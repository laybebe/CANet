import os
import sys
import pdb
import math
import torch
from PIL import Image
import cv2
import numpy as np
import random
import torch.nn.functional as F
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
from torchvision import transforms
ToPILImage = transforms.ToPILImage()
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def average(self):
        return self.avg

    def value(self):
        return self.val

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, difficult_examples=True):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())
        self.filenames = []

    def add(self, output, target, filename):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

        self.filenames += filename  # record filenames

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()
        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(
                scores, targets, self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):

        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        precision_at_i /= pos_count
        return precision_at_i

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        scores = self.scores.cpu().numpy()
        targets = self.targets.clone().cpu().numpy()
        targets[targets == -1] = 0
        return self.evaluation(scores, targets)

    def overall_topk(self, k):
        targets = self.targets.clone().cpu().numpy()
        targets[targets == -1] = 0
        n, c = self.scores.size()
        scores = np.zeros((n, c)) - 1
        index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
        tmp = self.scores.cpu().numpy()
        for i in range(n):
            for ind in index[i]:
                scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
        return self.evaluation(scores, targets)

    def evaluation(self, scores_, targets_):
        n, n_class = scores_.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            scores = scores_[:, k]
            targets = targets_[:, k]
            targets[targets == -1] = 0
            Ng[k] = np.sum(targets == 1)
            Np[k] = np.sum(scores >= 0)  # sigmoid函数，x为0时，y值为0.5
            Nc[k] = np.sum(targets * (scores >= 0))
        Np[Np == 0] = 1
        OP = np.sum(Nc) / np.sum(Np)
        OR = np.sum(Nc) / np.sum(Ng)
        OF1 = (2 * OP * OR) / (OP + OR)

        CP = np.sum(Nc / Np) / n_class
        CR = np.sum(Nc / Ng) / n_class
        CF1 = (2 * CP * CR) / (CP + CR)
        return OP, OR, OF1, CP, CR, CF1

def generate_heatmap(attention_maps):
    heat_attention_maps = []
    attention_maps = torch.sqrt(attention_maps.cpu() / attention_maps.max().item())
    heat_attention_maps.append((attention_maps >= 0.5).float())  # R
    heat_attention_maps.append(attention_maps *0)#(attention_maps < 0.5).float() + \
                               #(1. - attention_maps) * (attention_maps >= 0.5).float())  # G
    heat_attention_maps.append(attention_maps *0) #(1. - attention_maps)  # B
    return torch.stack(heat_attention_maps, dim=0),attention_maps

def visuaization(attention_maps, input, image_name):
    # reshape attention maps
    attention_maps = F.upsample_bilinear(
        attention_maps, size=(256,256))
    input = F.upsample_bilinear(input, size=(256, 256))
    attention_maps=attention_maps.view(attention_maps.size(0),20,10,attention_maps.size(2),attention_maps.size(3))
    attention_maps=attention_maps.mean(dim=2)
    
    # raw_image, heat_attention, raw_attention
    image_name=image_name[0]
    raw_image = input.cpu() * std + mean
    for i in range(attention_maps.size(1)):
        heat_attention_maps,attention_maps_temp=generate_heatmap(attention_maps[0,i,...])
        heat_attention_image = raw_image*(heat_attention_maps==1) * 0.4 + heat_attention_maps * 0.6+raw_image*(heat_attention_maps!=1)
        raw_attention_image = raw_image * attention_maps_temp
        rimg = ToPILImage(raw_image[0])
        raimg = ToPILImage(raw_attention_image[0])
        haimg = ToPILImage(heat_attention_image[0])
        # attention_maps_temp[attention_maps_temp>0.5]=1
        attention_map=ToPILImage(attention_maps_temp)
        im_gray =np.asarray(attention_map)
        img_color=cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
        img_c = Image.fromarray(cv2.cvtColor(img_color,cv2.COLOR_BGR2RGB))
        blend_img = Image.blend(img_c,rimg,0.5)
        rimg.save(f"/home/lpj/code/ADD-GCN/visual_resultvoc07V4/{image_name}_raw.jpg")
        raimg.save(f"/home/lpj/code/ADD-GCN/visual_resultvoc07V4/{image_name}_raw_atten_{i}.jpg")
        haimg.save(f"/home/lpj/code/ADD-GCN/visual_resultvoc07V4/{image_name}_heat_atten_{i}.jpg")
        blend_img.save(f"/home/lpj/code/ADD-GCN/visual_resultvoc07V4/{image_name}_blend_atten_{i}.jpg")
        # rimg.save(f"/home/lpj/code/ADD-GCN/visual_resultcoco/raw_{image_name}")
        # raimg.save(f"/home/lpj/code/ADD-GCN/visual_resultcoco/raw_atten_{i}_{image_name}" )
        # haimg.save(f"/home/lpj/code/ADD-GCN/visual_resultcoco/heat_atten_{i}_{image_name}")