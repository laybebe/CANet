import numpy as np


def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()


class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01


"""
functions used to calculate the metrics for multi-label classification
cmap=mAP, emap=MiAP
"""


def cemap_cal(y_pred, y_true):
    '''
    y_true: -1 negative; 0 difficult_examples; 1 positive.
    '''
    nTest = y_true.shape[0]
    nLabel = y_true.shape[1]
    ap = np.zeros(nTest)
    for i in range(0, nTest):
        R = np.sum(y_true[i, :] == 1)
        for j in range(0, nLabel):
            if y_true[i, j] == 1:
                r = np.sum(y_pred[i, np.nonzero(
                    y_true[i, :] != 0)] >= y_pred[i, j])
                rb = np.sum(y_pred[i, np.nonzero(
                    y_true[i, :] == 1)] >= y_pred[i, j])

                ap[i] = ap[i] + rb/(r*1.0)
        ap[i] = ap[i]/R
    emap = np.nanmean(ap)

    ap = np.zeros(nLabel)
    for i in range(0, nLabel):
        R = np.sum(y_true[:, i] == 1)
        for j in range(0, nTest):
            if y_true[j, i] == 1:
                r = np.sum(
                    y_pred[np.nonzero(y_true[:, i] != 0), i] >= y_pred[j, i])
                rb = np.sum(
                    y_pred[np.nonzero(y_true[:, i] == 1), i] >= y_pred[j, i])
                ap[i] = ap[i] + rb/(r*1.0)
        ap[i] = ap[i]/R
    cmap = np.nanmean(ap)

    return cmap, emap


def prf_cal(y_pred, y_true, k):
    """
    function to calculate top-k precision/recall/f1-score
    y_true: 0 1
    """
    GT = np.sum(y_true[y_true == 1.])
    instance_num = y_true.shape[0]
    prediction_num = instance_num*k

    sort_indices = np.argsort(y_pred)
    sort_indices = sort_indices[:, ::-1]
    static_indices = np.indices(sort_indices.shape)
    sorted_annotation = y_true[static_indices[0], sort_indices]
    top_k_annotation = sorted_annotation[:, 0:k]
    TP = np.sum(top_k_annotation[top_k_annotation == 1.])
    recall = TP/GT
    precision = TP/prediction_num
    f1 = 2.*recall*precision/(recall+precision)
    return precision, recall, f1


def cemap_cal_old(y_pred, y_true):
    """
    function to calculate C-MAP (mAP) and E-MAP
    y_true: 0 1
    """
    nTest = y_true.shape[0]
    nLabel = y_true.shape[1]
    ap = np.zeros(nTest)
    for i in range(0, nTest):
        R = np.sum(y_true[i, :])
        for j in range(0, nLabel):
            if y_true[i, j] == 1:
                r = np.sum(y_pred[i, :] >= y_pred[i, j])
                rb = np.sum(y_pred[i, np.nonzero(
                    y_true[i, :])] >= y_pred[i, j])
                ap[i] = ap[i] + rb/(r*1.0)
        ap[i] = ap[i]/R
    emap = np.nanmean(ap)

    ap = np.zeros(nLabel)
    for i in range(0, nLabel):
        R = np.sum(y_true[:, i])
        for j in range(0, nTest):
            if y_true[j, i] == 1:
                r = np.sum(y_pred[:, i] >= y_pred[j, i]) 
                rb = np.sum(
                    y_pred[np.nonzero(y_true[:, i]), i] >= y_pred[j, i])
                ap[i] = ap[i] + rb/(r*1.0)
        ap[i] = ap[i]/R
    cmap = np.nanmean(ap)

    return cmap, emap
