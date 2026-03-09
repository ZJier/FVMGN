# -*- coding: utf-8 -*-
import os
import torch
import random
import scipy
import mat73
import spectral
import torch.utils
import numpy as np
from tqdm import tqdm
import torch.utils.data
from scipy.linalg import sqrtm
from utils_RS import open_file
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve
from scipy.io import loadmat, savemat
from skimage.transform import resize


DATASETS_CONFIG = {
    'MUUFLHSI': {"urls": [],}, 'MUUFLLiDAR': {"urls": [],}, 'TrentoHSI': {"urls": [],}, 'TrentoLiDAR': {"urls": [],}, 
    'SubAugMSI1': {"urls": [],}, 'SubAugMSI2': {"urls": [],}, 'SubAugMSI3': {"urls": [],}, 'SubAugMSI4': {"urls": [],}, 
    'SubAugSAR1': {"urls": [],}, 'SubAugSAR2': {"urls": [],}, 'SubAugSAR3': {"urls": [],}, 'SubAugSAR4': {"urls": [],},
    'SubBerMSI1': {"urls": [],}, 'SubBerMSI2': {"urls": [],}, 'SubBerMSI3': {"urls": [],}, 'SubBerMSI4': {"urls": [],},
    'SubBerSAR1': {"urls": [],}, 'SubBerSAR2': {"urls": [],}, 'SubBerSAR3': {"urls": [],}, 'SubBerSAR4': {"urls": [],},
    'HU13HSI': {"urls": [],}, 'HU13LiDAR': {"urls": [],}, 'HU18HSI': {"urls": [],}, 'HU18LiDAR': {"urls": [],}, 
    }


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

def capture_hsi(path):
    data = scipy.io.loadmat(path)
    keys = list(data.keys())
    key_to_print = keys[3]
    array = data[key_to_print]
    return array['Data'][0][0]

def capture_lidar(path):
    data = scipy.io.loadmat(path)
    keys = list(data.keys())
    key_to_print = keys[3]
    array = data[key_to_print]
    return array['Lidar'][0][0][0][0][0][0][2]

def capture_gt(path):
    data = scipy.io.loadmat(path)
    keys = list(data.keys())
    key_to_print = keys[3]
    array = data[key_to_print]
    gt_array = array['sceneLabels'][0][0][0][0][6]
    gt_array = np.where(gt_array == -1, 0, gt_array)
    return gt_array

def get_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG):
    palette = None
    
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    dataset = datasets[dataset_name]

    folder = target_folder# + datasets[dataset_name].get('folder', dataset_name + '/')
    if dataset.get('download', False):
        # Download the dataset if is not present
        if not os.path.isdir(folder):
            os.mkdir(folder)
        for url in datasets[dataset_name]['urls']:
            # download the files
            filename = url.split('/')[-1]
            if not os.path.exists(folder + filename):
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc="Downloading {}".format(filename)) as t:
                    urlretrieve(url, filename=folder + filename, reporthook=t.update_to)
    elif not os.path.isdir(folder):
       print("WARNING: {} is not downloadable.".format(dataset_name))

    if dataset_name == 'MUUFLHSI':
        img = capture_hsi('./MS_Dataset/MUUFL/muufl_gulfport_campus_1_hsi_220_label.mat')
        gt = loadmat('./MS_Dataset/MUUFL/MUUFL_3gt.mat')['MUUFLGT']
        rgb_bands = (20, 15, 5)

        # [m, n, l] = img.shape
        # for i in range(l):
        #     minimal = img[:, :, i].min()
        #     maximal = img[:, :, i].max()
        #     img[:, :, i] = (img[:, :, i] - minimal) / (maximal - minimal)

        label_values = ["Trees", "Roads", "Buildings"]
        label_queue = {"Trees": ['Trees are usually next to roads and sidewalks, displaying rich green hues in the near infrared spectrum.', 
                                 'Trees are usually taller than other objects, except for buildings, clearly distinguishable based on their elevation and shape.'],
                    "Roads": ['Roads are usually next to sidewalks, buildings, and trees, appearing gray and brown in the visible spectrum, with a uniform and smooth texture.', 
                             'Roads are characterized by flattened surfaces and consistent elevations, which contrast sharply with the surrounding trees.'], 
                    "Buildings": ['Buildings are usually relatively regular large areas with high spectral reflectance in the visible and near infrared bands.', 
                                  'Buildings usually have higher elevations and more regular shapes, showcasing their vertical structures and varying heights across urban landscapes.']}
        ignored_labels = [0]
    
    elif dataset_name == 'MUUFLLiDAR':
        img = capture_lidar('./MS_Dataset/MUUFL/muufl_gulfport_campus_1_hsi_220_label.mat')
        gt = loadmat('./MS_Dataset/MUUFL/MUUFL_3gt.mat')['MUUFLGT']
        rgb_bands = (20, 15, 5)

        # [m, n, l] = img.shape
        # for i in range(l):
        #     minimal = img[:, :, i].min()
        #     maximal = img[:, :, i].max()
        #     img[:, :, i] = (img[:, :, i] - minimal) / (maximal - minimal)

        label_values = ["Trees", "Roads", "Buildings"]
        label_queue = {"Trees": ['Trees are usually next to roads and sidewalks, displaying rich green hues in the near infrared spectrum.', 
                                 'Trees are usually taller than other objects, except for buildings, clearly distinguishable based on their elevation and shape.'],
                    "Roads": ['Roads are usually next to sidewalks, buildings, and trees, appearing gray and brown in the visible spectrum, with a uniform and smooth texture.', 
                             'Roads are characterized by flattened surfaces and consistent elevations, which contrast sharply with the surrounding trees.'], 
                    "Buildings": ['Buildings are usually relatively regular large areas with high spectral reflectance in the visible and near infrared bands.', 
                                  'Buildings usually have higher elevations and more regular shapes, showcasing their vertical structures and varying heights across urban landscapes.']}
        ignored_labels = [0]
    
    elif dataset_name == 'TrentoHSI': 
        img = np.asarray(open_file('./MS_Dataset/Trento/HSI_Trento.mat')['hsi_trento'])
        gt = loadmat('./MS_Dataset/Trento/Trento_3wgt.mat')['TrentoGT']
        rgb_bands = (20, 15, 10)

        label_values = ["Trees", "Roads", "Buildings"]
        label_queue = {"Trees": ['Trees are usually next to roads and sidewalks, displaying rich green hues in the near infrared spectrum.', 
                                 'Trees are usually taller than other objects, except for buildings, clearly distinguishable based on their elevation and shape.'],
                    "Roads": ['Roads are usually next to sidewalks, buildings, and trees, appearing gray and brown in the visible spectrum, with a uniform and smooth texture.', 
                             'Roads are characterized by flattened surfaces and consistent elevations, which contrast sharply with the surrounding trees.'], 
                    "Buildings": ['Buildings are usually relatively regular large areas with high spectral reflectance in the visible and near infrared bands.', 
                                  'Buildings usually have higher elevations and more regular shapes, showcasing their vertical structures and varying heights across urban landscapes.']}
        ignored_labels = [0]
    
    elif dataset_name == 'TrentoLiDAR':
        img = np.asarray(open_file('./MS_Dataset/Trento/Lidar1_Trento.mat')['lidar1_trento'])
        img = np.expand_dims(img, axis=2)
        gt = loadmat('./MS_Dataset/Trento/Trento_3wgt.mat')['TrentoGT']
        rgb_bands = (0, 1, 2)

        label_values = ["Trees", "Roads", "Buildings"]
        label_queue = {"Trees": ['Trees are usually next to roads and sidewalks, displaying rich green hues in the near infrared spectrum.', 
                                 'Trees are usually taller than other objects, except for buildings, clearly distinguishable based on their elevation and shape.'],
                    "Roads": ['Roads are usually next to sidewalks, buildings, and trees, appearing gray and brown in the visible spectrum, with a uniform and smooth texture.', 
                             'Roads are characterized by flattened surfaces and consistent elevations, which contrast sharply with the surrounding trees.'], 
                    "Buildings": ['Buildings are usually relatively regular large areas with high spectral reflectance in the visible and near infrared bands.', 
                                  'Buildings usually have higher elevations and more regular shapes, showcasing their vertical structures and varying heights across urban landscapes.']}
        ignored_labels = [0]

    elif dataset_name == 'HU13HSI': 
        img = loadmat('./MS_Dataset/HU2013/hu13hsi_new_hs_lr.mat')['hu13hsi_new']
        gt = loadmat('./MS_Dataset/HU2013/HU13_GT3_new.mat')['hu13gt_new']
        rgb_bands = (64, 43, 20)

        label_values = ["Trees", "Roads", "Buildings"]
        label_queue = {"Trees": ['Trees are usually next to roads and sidewalks, displaying rich green hues in the near infrared spectrum.', 
                                 'Trees are usually taller than other objects, except for buildings, clearly distinguishable based on their elevation and shape.'],
                    "Roads": ['Roads are usually next to sidewalks, buildings, and trees, appearing gray and brown in the visible spectrum, with a uniform and smooth texture.', 
                             'Roads are characterized by flattened surfaces and consistent elevations, which contrast sharply with the surrounding trees.'], 
                    "Buildings": ['Buildings are usually relatively regular large areas with high spectral reflectance in the visible and near infrared bands.', 
                                  'Buildings usually have higher elevations and more regular shapes, showcasing their vertical structures and varying heights across urban landscapes.']}
        ignored_labels = [0]
    
    elif dataset_name == 'HU13LiDAR': 
        img = np.expand_dims(loadmat('./MS_Dataset/HU2013/hu13lidar_new.mat')['hu13lidar_new'], axis=2)
        gt = loadmat('./MS_Dataset/HU2013/HU13_GT3_new.mat')['hu13gt_new']
        rgb_bands = (20, 15, 10)
        label_values = ["Trees", "Roads", "Buildings"]
        label_queue = {"Trees": ['Trees are usually next to roads and sidewalks, displaying rich green hues in the near infrared spectrum.', 
                                 'Trees are usually taller than other objects, except for buildings, clearly distinguishable based on their elevation and shape.'],
                    "Roads": ['Roads are usually next to sidewalks, buildings, and trees, appearing gray and brown in the visible spectrum, with a uniform and smooth texture.', 
                             'Roads are characterized by flattened surfaces and consistent elevations, which contrast sharply with the surrounding trees.'], 
                    "Buildings": ['Buildings are usually relatively regular large areas with high spectral reflectance in the visible and near infrared bands.', 
                                  'Buildings usually have higher elevations and more regular shapes, showcasing their vertical structures and varying heights across urban landscapes.']}
        ignored_labels = [0]

    else:
        print("This dataset is missing.")

    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
       print("Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)
    ignored_labels = list(set(ignored_labels))
    # Normalization
    img = np.asarray(img, dtype='float32')
    
    m, n, d = img.shape[0], img.shape[1], img.shape[2]
    img= img.reshape((m * n, -1))
    img = img / img.max()
    img_temp = np.sqrt(np.asarray((img**2).sum(1)))
    img_temp = np.expand_dims(img_temp, axis = 1)
    img_temp = img_temp.repeat(d, axis = 1)
    img_temp[img_temp == 0] = 1
    img = img / img_temp
    img = np.reshape(img,(m, n, -1))

    # return img, gt, label_values, ignored_labels, rgb_bands, palette
    return img, gt, label_values, label_queue, ignored_labels, rgb_bands, palette


class HyperX(torch.utils.data.Dataset):
    def __init__(self, data, gt, transform=None, **hyperparams):
        super(HyperX, self).__init__()
        self.transform = transform
        self.data = data
        self.label = gt
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.radiation_augmentation = hyperparams['radiation_augmentation'] 
        self.mixture_augmentation = hyperparams['mixture_augmentation'] 
        self.center_pixel = hyperparams['center_pixel']
        supervision = hyperparams['supervision']
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x,y] for x,y in self.indices]
        all_indices = self.indices
        
        # state = np.random.get_state()
        # np.random.shuffle(self.indices)
        # np.random.set_state(state)
        # np.random.shuffle(self.labels)

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc = 0., scale = 1.0, size = data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size = 2)
        noise = np.random.normal(loc = 0., scale = 1.0, size = data.shape)
        data2 = np.zeros_like(data)
        for  idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.flip_augmentation and self.patch_size > 1 and np.random.random() < 0.5:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.5: 
                data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.5: 
                data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]
        else:
            label = self.labels[i]
            
        # Add a fourth dimension for 3D CNN
        # if self.patch_size > 1:
        #     # Make 4D data ((Batch x) Planes x Channels x Width x Height)
        #     data = data.unsqueeze(0)
        # plt.imshow(data[[10,23,23],:,:].permute(1,2,0))
        # plt.show()
        return data, label

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.data, self.label = next(self.loader)

        except StopIteration:
            self.next_input = None

            return
        with torch.cuda.stream(self.stream):
            self.data = self.data.cuda(non_blocking=True)
            self.label = self.label.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.data
        label = self.label

        self.preload()
        return data, label
