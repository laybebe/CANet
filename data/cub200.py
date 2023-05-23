from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import csv
import numpy as np
object_categories=['1','2','3','4','5','6','7','8','9','10',
                    '11','12','13','14','15','16','17','18','19','20',
                    '21','22','23','24','25','26','27','28','29','30',
                    '31','32','33','34','35','36','37','38','39','40',
                    '41','42','43','44','45','46','47','48','49','50',
                    '51','52','53','54','55','56','57','58','59','60',
                    '61','62','63','64','65','66','67','68','69','70',
                    '71','72','73','74','75','76','77','78','79','80',
                    '81','82','83','84','85','86','87','88','89','90',
                    '91','92','93','94','95','96','97','98','99','100',
                    '101','102','103','104','105','106','107','108','109','110', 
                    '111','112','113','114','115','116','117','118','119','120', 
                    '121','122','123','124','125','126','127','128','129','130', 
                    '131','132','133','134','135','136','137','138','139','140', 
                    '141','142','143','144','145','146','147','148','149','150', 
                    '151','152','153','154','155','156','157','158','159','160', 
                    '161','162','163','164','165','166','167','168','169','170', 
                    '171','172','173','174','175','176','177','178','179','180',
                    '181','182','183','184','185','186','187','188','189','190', 
                    '191','192','193','194','195','196','197','198','199','200'                    
                    ]

def search_file(data_path, target):  # serch file
    file_list = []
    for dirpath, dirnames, filenames in os.walk(data_path):
        for f in filenames:
            ext = os.path.splitext(f)[1]
            if ext in target:
                file_list.append(os.path.join(dirpath, f))
    return file_list

def write_object_labels_csv(file_csv, path_label, path_split,phase):
    # write a csv file   
    print('[dataset] write file %s' % file_csv)
    if phase=="train":
        flag="1"
    else:
        flag="0"
    with open(path_label) as label_f:
        path_label_list=label_f.readlines()
    with open(path_split) as split_f:
        path_split_list=split_f.readlines()  
    with open(file_csv, 'w') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(object_categories)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for  k,label in  enumerate(path_label_list):
            split=path_split_list[k].strip('\n').split(' ')
            label=label.strip('\n').split(' ')
            name=label[0]
            if flag==split[1]:
                example = {'name': name}
                labels=np.zeros(200)-1
                labels[int(label[1])-1]=1
                for i in range(200):
                    example[fieldnames[i + 1]] = int(labels[i])
                writer.writerow(example)
    csvfile.close()


def read_object_labels_csv(file, id_image,header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    id_image_dict={}
    with open(id_image,'r') as id_f:
        data=id_f.readlines()
        for id_map in data:
            id=id_map.strip('\n').split(' ')
            id_image_dict[id[0]]=id[1]
    id_f.close()
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                name = id_image_dict[name]
                labels = torch.from_numpy((np.asarray(row[1:num_categories + 1])).astype(np.float32))
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images

class CUB200(Dataset):
    def __init__(self, root, phase, transform=None):
        self.root = os.path.abspath(root)
        self.path_images = os.path.join(self.root, 'images')
        self.phase = phase
        self.transform = transform
        # define path of csv file
        path_csv = os.path.join(self.root, 'files','CUB200')
        # define filename of csv file
        file_csv = os.path.join(path_csv, 'classification_' + phase + '.csv')
        id_image=os.path.join(self.root, 'images.txt')
        path_label=os.path.join(self.root, 'image_class_labels.txt')
        path_split=os.path.join(self.root, 'train_test_split.txt')
        # create the csv file if necessary
        if not os.path.exists(file_csv):
            if not os.path.exists(path_csv):  # create dir if necessary
                os.makedirs(path_csv)
            write_object_labels_csv(file_csv,path_label,path_split,phase)
        self.images = read_object_labels_csv(file_csv,id_image)
        self.classes = object_categories
        print('[dataset] CUB 200 classification phase={} number of classes={}  number of images={}'.format(phase, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        filename, target = self.images[index]
        img = Image.open(os.path.join(self.path_images, filename)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        data = {'image':img, 'name': filename, 'target': target}
        return data


    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)

if __name__ == "__main__":

    phase="test"
    root='/home/lpj/datasets/CUB_200_2011'
    path_csv = os.path.join(root, 'files','CUB200')
    path_label=os.path.join(root, 'image_class_labels.txt')
    path_split=os.path.join(root, 'train_test_split.txt')
    file_csv = os.path.join(path_csv, 'classification_' + phase + '.csv')
    id_image=os.path.join(root, 'images.txt')
    # create the csv file if necessary
    if not os.path.exists(file_csv):
        if not os.path.exists(path_csv):  # create dir if necessary
            os.makedirs(path_csv)
        write_object_labels_csv(file_csv,path_label,path_split,phase)

    images = read_object_labels_csv(file_csv,id_image)
    pass
    