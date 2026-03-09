from __future__ import print_function
import os
import math
import clip
import time
import tqdm
import torch
import random
import argparse
import datetime
import numpy as np
import pandas as pd
import scipy.io as io
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
from datetime import datetime
from models import FVMGN
import torch.utils.data as data
import torch.nn.functional as F
# from torchinfo import summary
from torchsummary import summary
import matplotlib
from scipy.io import loadmat, savemat
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from thop import (profile, clever_format)
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report
from datasets import get_dataset, HyperX
# from EVO_MM import loss_cluster, ClusterAlignmentLoss
from utils_RS import (sample_gt, metrics, get_device, seed_worker, pad_gt, pad_img, 
                       show_results, tensor2img, write_results, classification_map, save_figs)


parser = argparse.ArgumentParser(description='Cross-modal Domain Adaptation')
parser.add_argument('--model', type=str, default='FVMGN', 
                    help="Available:\n"
                    "ViT, "
                    "CNN, "
                    "FVMGN, ")
# Data options 
parser.add_argument('--runs', type=int, default=20, help="Number of runs (default 5)")
parser.add_argument('--top_k', type=int, default=10, help="Number of runs (default 5)")
parser.add_argument('--save_path', type=str, default="./results/", help='the path to save the model')
# MUUFLHSI, MUUFLLiDAR, TrentoHSI, TrentoLiDAR, HU13HSI, HU13LiDAR
parser.add_argument('--data_path', type=str, default='D:/Working/My_Code/Datasets/MSDataset/', help='the path to load the data')
parser.add_argument('--source_name', type=str, default='MUUFLHSI', help='the name of the source dir')
parser.add_argument('--source_name2', type=str, default='MUUFLLiDAR', help='the name of the source dir')
parser.add_argument('--target_name', type=str, default='TrentoHSI', help='.the name of the test dir')
parser.add_argument('--target_name2', type=str, default='TrentoLiDAR', help='.the name of the test dir')
parser.add_argument('--cuda', type=int, default=0, help="Specify CUDA device (defaults to -1, which learns on CPU)")
# Training options
group_train = parser.add_argument_group('Training')
parser.add_argument('--seed', type=int, default=3667, metavar='S', help='random seed') # 3667
group_train.add_argument('--patch_size', type=int, default=11, help="Spatial neighbourhood")
group_train.add_argument('--lr', type=float, default=0.001, help="Learning rate")
parser.add_argument('--num_epoch', type=int, default=20, help='the number of epoch') # 20
group_train.add_argument('--batch_size', type=int, default=128, help="Batch size (256)")
parser.add_argument('--re_ratio', type=int, default=5, help='multiple of data augmentation')
parser.add_argument('--training_sample_ratio', type=float, default=0.1, help='training sample ratio')
# Other Super-parameters
group_train.add_argument('--lambda_1', type=float, default=1e+0, help="Regularization parameter, balancing the alignment loss.")
group_train.add_argument('--alpha', type=float, default=0.3, help="Regularization parameter, controlling the contribution of both coarse-and fine-grained linguistic features.")
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.5)')
group_train.add_argument('--class_balancing', action='store_true', help="Inverse median frequency class balancing (default = False)")
group_train.add_argument('--test_stride', type=int, default=0, help="Sliding window step stride during inference (default = 1)")
parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--l2_decay', type=float, default=1e-4, help='the L2 weight decay')
parser.add_argument('--num_trials', type=int, default=1, help='the number of epoch')
parser.add_argument('--fine_text', type=int, default=2, help="0: Non; 1: One; 2: Two.")
# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=False, help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true', default=False, help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true', default=False, help="Random mixes between spectra")
# Visualization
parser.add_argument('--with_exploration', default=True, action='store_true', help="See data exploration visualization")
# All Loading
args = parser.parse_args()

DEVICE = get_device(args.cuda)
N_RUNS = args.runs
F_text = args.fine_text
DATASET = args.source_name
SOURCE_1 = args.source_name
SOURCE_2 = args.source_name2
TARGET_1 = args.target_name
TARGET_2 = args.target_name2
MODEL = args.model
BATCHSIZE = args.batch_size
PATCHSIZE = args.patch_size
TOP_K = args.top_k

def applyPCA(X, numComponents = 30):
    # set_deterministic(2)
    random.seed(2)
    np.random.seed(2)
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca

def train(epoch, model, num_epoch, label_name, label_queue):
    fea_hsi_lists, fea_other_lists, label_lists = [], [], []
    # ===========================================================================
    # LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch - 1) / num_epoch), 0.75)
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # if (epoch - 1) % 10 == 0: 
    #     print('learning rate{: .4f}'.format(LEARNING_RATE))
    # ===========================================================================
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, epoch)

    CNN_correct= 0
    iter_source = iter(train_loader)
    num_iter = len_src_loader

    for i in range(1, num_iter):
        model.train()
        data_src, label_src = iter_source.__next__()
        data_src, label_src = data_src.to(DEVICE), label_src.to(DEVICE)
        label_src = label_src - 1
        optimizer.zero_grad()
        label_src = label_src.long()

        text = torch.cat([clip.tokenize(f'A remote sensing image of {label_name[k]}').to(k.device) for k in label_src])
        if F_text == 0: 
            loss_time, loss_feq, img_prob, fea_hsi, fea_oth = model(data_src, text, label_src, 
                                                            text_queue_1=None, text_queue_2=None)
        elif F_text == 1: 
            text_queue_1 = [label_queue[label_name[k]][0] for k in label_src]
            text_queue_1 = torch.cat([clip.tokenize(k).to(text.device) for k in text_queue_1])
            loss_time, loss_feq, img_prob, fea_hsi, fea_oth = model(data_src, text, label_src, 
                                                            text_queue_1=text_queue_1, text_queue_2=None)
            # text_queue_2 = [label_queue[label_name[k]][1] for k in label_src]
            # text_queue_2 = torch.cat([clip.tokenize(k).to(text.device) for k in text_queue_2])
            # loss_time, loss_feq, img_prob = model(data_src, text, label_src, 
            #                                                 text_queue_1=None, text_queue_2=text_queue_2)
        elif F_text == 2: 
            text_queue_1 = [label_queue[label_name[k]][0] for k in label_src]
            text_queue_2 = [label_queue[label_name[k]][1] for k in label_src]
            text_queue_1 = torch.cat([clip.tokenize(k).to(text.device) for k in text_queue_1])
            text_queue_2 = torch.cat([clip.tokenize(k).to(text.device) for k in text_queue_2])
            loss_time, loss_feq, img_prob, fea_hsi, fea_oth = model(data_src, text, label_src, 
                                                            text_queue_1=text_queue_1, text_queue_2=text_queue_2)

        label_src_pred_hsi = img_prob[:, :num_classes]
        label_src_pred_other = img_prob[:, num_classes:]
        # ==============================================================================
        label_src_preds = torch.stack([label_src_pred_hsi, label_src_pred_other], dim=1)
        label_src_pred, _ = torch.max(label_src_preds, dim=1)
        # F.nll_loss
        loss_cls_hsi = F.nll_loss(F.log_softmax(label_src_pred_hsi, dim=1), label_src.long())
        loss_cls_other = F.nll_loss(F.log_softmax(label_src_pred_other, dim=1), label_src.long())
        loss_cls = (loss_cls_hsi + loss_cls_other) / 2
        # ===============================================================================
        loss = loss_cls + args.lambda_1 * ((1 - args.alpha) * loss_time + args.alpha * loss_feq)
        loss.backward()
        
        optimizer.step()
        pred = label_src_pred.data.max(1)[1]
        # ==========================Plot pred features t-SNE==========================
        label_lists.append(label_src.cpu().numpy())
        fea_hsi_lists.append(fea_hsi.clone().detach().cpu().numpy())
        fea_other_lists.append(fea_oth.clone().detach().cpu().numpy())
        # =============================================================================
        CNN_correct += pred.eq(label_src.data.view_as(pred)).cpu().sum()

        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format( epoch, i * len(data_src), len_src_dataset, 100. * i / len_src_loader))
            if F_text == 0: 
                print('loss: {:.6f},  loss_cls: {:.6f}, loss_time: {:.6f}, loss_feq: None'.format(
                                                loss.item(), loss_cls.item(), loss_time.item()))
            else: 
                print('loss: {:.6f},  loss_cls: {:.6f}, loss_time: {:.6f}, loss_feq: {:.6f}'.format(
                                                loss.item(), loss_cls.item(), loss_time.item(), loss_feq.item()))
    scheduler.step()
    CCN_acc = CNN_correct.item() / len_src_dataset
    print('[epoch: {:4}]  Train Accuracy: {:.4f} | train sample number: {:6}'.format(epoch, CCN_acc, len_src_dataset))

    return model, CCN_acc, label_lists, fea_hsi_lists, fea_other_lists


def test(model, label_name):
    model.eval()
    loss = 0
    correct = 0
    loss_time = 0
    pred_list, label_list = [], []
    fea_hsi_list, fea_other_list = [], []

    with torch.no_grad():
        # if you want to plt cls map (infer the whole img), you should use all_loaders, rather than test_loader.
        for data, label in test_loader: 
            data, label = data.to(DEVICE), label.to(DEVICE)
            label = label - 1
            label = label.long()
            text = torch.cat([clip.tokenize(f'A remote sensing image of {label_name[k]}').to(k.device) for k in label])
            loss_time_, label_src_pred, fea_hsi, fea_other = model(data, text, label)
            pred = label_src_pred.data.max(1)[1]
            pred_list.append(pred.cpu().numpy())
            label_list.append(label.cpu().numpy())
            # ==========================Plot pred features t-SNE==========================
            fea_hsi_list.append(fea_hsi.cpu().numpy())
            fea_other_list.append(fea_other.cpu().numpy())
            # =============================================================================
            loss += F.nll_loss(F.log_softmax(label_src_pred, dim = 1), label.long()).item()
            loss_time += loss_time_.item()
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        loss /= len_tar_loader
        loss_time /= len_tar_loader
        print('Average test loss: {:.4f}, loss clip: {:.4f}, test Accuracy: {}/{} ({:.2f}%), | test sample number: {:6}\n'.format(
            loss, loss_time, correct, len_tar_dataset, 100. * correct / len_tar_dataset, len_tar_dataset))

    return correct, correct.item() / len_tar_dataset, pred_list, label_list, fea_hsi_list, fea_other_list


if __name__ == '__main__':
    results_all = []
    for run in range(N_RUNS): 
        print('Run Times:', run + 1)
        args.save_path = os.path.join(args.save_path)
        # args.save_path = os.path.join(args.save_path, args.source_name+'to'+args.target_name)
        acc_test_list, acc_maxval_test_list = np.zeros([args.num_trials, 1]), np.zeros([args.num_trials, 1])
        seed_worker((args.seed + run) if args.source_name == 'TrentoHSI' else (args.seed - 9 * run))
        
        img_src_1, gt_src, LABEL_VALUES_src, LABEL_QUEUE, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.source_name, args.data_path)
        img_src_2, _, _, _, _, _, _ = get_dataset(args.source_name2, args.data_path)
        img_tar_1, gt_tar, LABEL_VALUES_tar, LABEL_QUEUE, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_name, args.data_path)
        img_tar_2, _, _, _, _, _, _ = get_dataset(args.target_name2,args.data_path)
        # =============================================================================================
        if F_text == 0: 
            LABEL_QUEUE = None
            print("No Fine Text.")
        elif F_text == 1: 
            print("Using 1 Fine Text.")
        elif F_text == 2: 
            print("Using 2 Fine Text.")
        # ====================================================================================================
        # Reduce HSI channels and Expand SAR/LiDAR/MSI channels
        if DATASET == 'MUUFLHSI': 
            img_src_1_data, pcas = applyPCA(img_src_1, numComponents = 30)
            img_src_1_know = np.load('./MS_Dataset/MUUFL/PCA30_t5_2_full.pkl.npy')
            img_src_1_all = np.concatenate((img_src_1_data, img_src_1_know), axis=2)
            padding_img_src = img_src_1_data.shape[2] - img_src_2.shape[2]
            img_src_2_data = np.pad(img_src_2, ((0, 0), (0, 0), (0, padding_img_src)), mode='reflect')
            img_src_2_know = np.load('./MS_Dataset/MUUFL/PCA2_t5_2_full.pkl.npy')
            img_src_2_know = np.pad(img_src_2_know, ((0, 0), (0, 0), (0, padding_img_src)), mode='reflect')
            img_src_2_all = np.concatenate((img_src_2_data, img_src_2_know), axis=2)
            img_src = np.concatenate((img_src_1_all, img_src_2_all), axis=2)
            # ===========================================================
            img_tar_1_data, pcas = applyPCA(img_tar_1, numComponents = 30)
            img_tar_1_know = np.load('./MS_Dataset/Trento/PCA30_t5_2_full.pkl.npy')
            img_tar_1_all = np.concatenate((img_tar_1_data, img_tar_1_know), axis=2)
            padding_img_tar = img_tar_1_data.shape[2] - img_tar_2.shape[2]
            img_tar_2_data = np.pad(img_tar_2, ((0, 0), (0, 0), (0, padding_img_tar)), mode='reflect')
            img_tar_2_know = np.load('./MS_Dataset/Trento/PCA1_t5_2_full.pkl.npy')
            img_tar_2_know = np.pad(img_tar_2_know, ((0, 0), (0, 0), (0, padding_img_tar)), mode='reflect')
            img_tar_2_all = np.concatenate((img_tar_2_data, img_tar_2_know), axis=2)
            img_tar = np.concatenate((img_tar_1_all, img_tar_2_all), axis=2)
            # ===========================================================
            # img_tar_1_data, pcas = applyPCA(img_tar_1, numComponents = 30)
            # # img_tar_1_know = np.load('./MS_Dataset/HU2013/Houston_K30_full_diff_e10000_t52.pkl.npy')
            # img_tar_1_know = loadmat('./MS_Dataset/HU2013/hu13hsi_k.mat')['hu13hsi_k']
            # img_tar_1_all = np.concatenate((img_tar_1_data, img_tar_1_know), axis=2)
            # padding_img_tar = img_tar_1_data.shape[2] - img_tar_2.shape[2]
            # img_tar_2_data = np.pad(img_tar_2, ((0, 0), (0, 0), (0, padding_img_tar)), mode='reflect')
            # img_tar_2_know = loadmat('./MS_Dataset/HU2013/hu13lidar_k.mat')['hu13lidar_k']
            # img_tar_2_know = np.pad(img_tar_2_know, ((0, 0), (0, 0), (0, padding_img_tar)), mode='reflect')
            # img_tar_2_all = np.concatenate((img_tar_2_data, img_tar_2_know), axis=2)
            # img_tar = np.concatenate((img_tar_1_all, img_tar_2_all), axis=2)
            # ===========================================================
        elif DATASET == 'TrentoHSI': 
            img_src_1_data, pcas = applyPCA(img_src_1, numComponents = 30)
            img_src_1_know = np.load('./MS_Dataset/Trento/PCA30_t5_2_full.pkl.npy')
            img_src_1_all = np.concatenate((img_src_1_data, img_src_1_know), axis=2)
            padding_img_src = img_src_1_data.shape[2] - img_src_2.shape[2]
            img_src_2_data = np.pad(img_src_2, ((0, 0), (0, 0), (0, padding_img_src)), mode='reflect')
            img_src_2_know = np.load('./MS_Dataset/Trento/PCA1_t5_2_full.pkl.npy')
            img_src_2_know = np.pad(img_src_2_know, ((0, 0), (0, 0), (0, padding_img_src)), mode='reflect')
            img_src_2_all = np.concatenate((img_src_2_data, img_src_2_know), axis=2)
            img_src = np.concatenate((img_src_1_all, img_src_2_all), axis=2)
            # ===========================================================
            img_tar_1_data, pcas = applyPCA(img_tar_1, numComponents = 30)
            img_tar_1_know = np.load('./MS_Dataset/MUUFL/PCA30_t5_2_full.pkl.npy')
            img_tar_1_all = np.concatenate((img_tar_1_data, img_tar_1_know), axis=2)
            padding_img_tar = img_tar_1_data.shape[2] - img_tar_2.shape[2]
            img_tar_2_data = np.pad(img_tar_2, ((0, 0), (0, 0), (0, padding_img_tar)), mode='reflect')
            img_tar_2_know = np.load('./MS_Dataset/MUUFL/PCA2_t5_2_full.pkl.npy')
            img_tar_2_know = np.pad(img_tar_2_know, ((0, 0), (0, 0), (0, padding_img_tar)), mode='reflect')
            img_tar_2_all = np.concatenate((img_tar_2_data, img_tar_2_know), axis=2)
            img_tar = np.concatenate((img_tar_1_all, img_tar_2_all), axis=2)
            # ===========================================================
            # img_tar_1_data, pcas = applyPCA(img_tar_1, numComponents = 30)
            # # # img_tar_1_know = np.load('./MS_Dataset/HU2013/Houston_K30_full_diff_e10000_t52.pkl.npy')
            # img_tar_1_know = loadmat('./MS_Dataset/HU2013/hu13hsi_k.mat')['hu13hsi_k']
            # img_tar_1_all = np.concatenate((img_tar_1_data, img_tar_1_know), axis=2)
            # padding_img_tar = img_tar_1_data.shape[2] - img_tar_2.shape[2]
            # img_tar_2_data = np.pad(img_tar_2, ((0, 0), (0, 0), (0, padding_img_tar)), mode='reflect')
            # img_tar_2_know = loadmat('./MS_Dataset/HU2013/hu13lidar_k.mat')['hu13lidar_k']
            # img_tar_2_know = np.pad(img_tar_2_know, ((0, 0), (0, 0), (0, padding_img_tar)), mode='reflect')
            # img_tar_2_all = np.concatenate((img_tar_2_data, img_tar_2_know), axis=2)
            # img_tar = np.concatenate((img_tar_1_all, img_tar_2_all), axis=2)
            # ===========================================================
        elif DATASET == 'HU13HSI': 
            # img_src_1_data, pcas = applyPCA(img_src_1, numComponents = 30)
            # padding_img_src = img_src_1_data.shape[2] - img_src_2.shape[2]
            # img_src_2_data = np.pad(img_src_2, ((0, 0), (0, 0), (0, padding_img_src)), mode='reflect')
            # img_src = np.concatenate((img_src_1_data, img_src_2_data), axis=2)
            # ===========================================================
            # img_tar_1_data, pcas = applyPCA(img_tar_1, numComponents = 30)
            # padding_img_tar = img_tar_1_data.shape[2] - img_tar_2.shape[2]
            # img_tar_2_data = np.pad(img_tar_2, ((0, 0), (0, 0), (0, padding_img_tar)), mode='reflect')
            # img_tar = np.concatenate((img_tar_1_data, img_tar_2_data), axis=2)
            # ===========================================================
            img_src_1_data, pcas = applyPCA(img_src_1, numComponents = 30)
            # img_src_1_know = np.load('./MS_Dataset/HU2013/Houston_K30_full_diff_e10000_t52.pkl.npy')
            img_src_1_know = loadmat('./MS_Dataset/HU2013/hu13hsi_k.mat')['hu13hsi_k']
            img_src_1_all = np.concatenate((img_src_1_data, img_src_1_know), axis=2)
            padding_img_src = img_src_1_data.shape[2] - img_src_2.shape[2]
            img_src_2_data = np.pad(img_src_2, ((0, 0), (0, 0), (0, padding_img_src)), mode='reflect')
            img_src_2_know = loadmat('./MS_Dataset/HU2013/hu13lidar_k.mat')['hu13lidar_k']
            img_src_2_know = np.pad(img_src_2_know, ((0, 0), (0, 0), (0, padding_img_src)), mode='reflect')
            img_src_2_all = np.concatenate((img_src_2_data, img_src_2_know), axis=2)
            img_src = np.concatenate((img_src_1_all, img_src_2_all), axis=2)
            # ===========================================================
            # img_tar_1_data, pcas = applyPCA(img_tar_1, numComponents = 30)
            # img_tar_1_know = np.load('./MS_Dataset/MUUFL/PCA30_t5_2_full.pkl.npy')
            # img_tar_1_all = np.concatenate((img_tar_1_data, img_tar_1_know), axis=2)
            # padding_img_tar = img_tar_1_data.shape[2] - img_tar_2.shape[2]
            # img_tar_2_data = np.pad(img_tar_2, ((0, 0), (0, 0), (0, padding_img_tar)), mode='reflect')
            # img_tar_2_know = np.load('./MS_Dataset/MUUFL/PCA2_t5_2_full.pkl.npy')
            # img_tar_2_know = np.pad(img_tar_2_know, ((0, 0), (0, 0), (0, padding_img_tar)), mode='reflect')
            # img_tar_2_all = np.concatenate((img_tar_2_data, img_tar_2_know), axis=2)
            # img_tar = np.concatenate((img_tar_1_all, img_tar_2_all), axis=2)
            # ===========================================================
            img_tar_1_data, pcas = applyPCA(img_tar_1, numComponents = 30)
            img_tar_1_know = np.load('./MS_Dataset/Trento/PCA30_t5_2_full.pkl.npy')
            img_tar_1_all = np.concatenate((img_tar_1_data, img_tar_1_know), axis=2)
            padding_img_tar = img_tar_1_data.shape[2] - img_tar_2.shape[2]
            img_tar_2_data = np.pad(img_tar_2, ((0, 0), (0, 0), (0, padding_img_tar)), mode='reflect')
            img_tar_2_know = np.load('./MS_Dataset/Trento/PCA1_t5_2_full.pkl.npy')
            img_tar_2_know = np.pad(img_tar_2_know, ((0, 0), (0, 0), (0, padding_img_tar)), mode='reflect')
            img_tar_2_all = np.concatenate((img_tar_2_data, img_tar_2_know), axis=2)
            img_tar = np.concatenate((img_tar_1_all, img_tar_2_all), axis=2)
            # ===========================================================
        else: 
            img_src_1, pcas = applyPCA(img_src_1, numComponents = 30)
            padding_img_src = img_src_1.shape[2] - img_src_2.shape[2]
            img_src_2 = np.pad(img_src_2, ((0, 0), (0, 0), (0, padding_img_src)), mode='reflect')
            img_src = np.concatenate((img_src_1, img_src_2), axis=2)

            img_tar_1, pcas = applyPCA(img_tar_1, numComponents = 30)
            padding_img_tar = img_tar_1.shape[2] - img_tar_2.shape[2]
            img_tar_2 = np.pad(img_tar_2, ((0, 0), (0, 0), (0, padding_img_tar)), mode='reflect')
            img_tar = np.concatenate((img_tar_1, img_tar_2), axis=2)
        print('Dimension (Cat Multi-source domain): ', img_src.shape)
        print('Dimension (Cat Multi-target domain): ', img_tar.shape)
        # ====================================================================================================
        sample_num_src = len(np.nonzero(gt_src)[0])
        sample_num_tar = len(np.nonzero(gt_tar)[0])
        training_sample_tar_ratio = args.training_sample_ratio * args.re_ratio * sample_num_src / sample_num_tar

        num_classes = gt_src.max()
        N_BANDS = img_src.shape[-1]
        hyperparams = vars(args)
        hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 
                        'device': DEVICE, 'center_pixel': False, 'supervision': 'full'})
        hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)
        # ===================================================================================================
        # try
        r = int(hyperparams['patch_size'] / 2) + 1
        img_src = np.pad(img_src,((r, r), (r, r), (0, 0)), 'symmetric')
        img_tar = np.pad(img_tar,((r, r), (r, r), (0, 0)), 'symmetric')
        gt_src = np.pad(gt_src,((r, r), (r, r)), 'constant', constant_values=(0, 0))
        gt_tar = np.pad(gt_tar,((r, r), (r, r)), 'constant', constant_values=(0, 0))
        # ===================================================================================================
        # try
        # img_src = pad_img(img_src, hyperparams['patch_size']//2).astype(np.float32)
        # img_tar = pad_img(img_tar, hyperparams['patch_size']//2).astype(np.float32)
        # gt_src = pad_gt(gt_src, hyperparams['patch_size']//2).astype(np.float32)
        # gt_tar = pad_gt(gt_tar, hyperparams['patch_size']//2).astype(np.float32)
        # ====================================================================================================
        train_gt_src, _, training_set, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
        test_gt_tar, _, tesing_set, _ = sample_gt(gt_tar, 1, mode='random')
        # ====================================================================================================
        img_src_con, train_gt_src_con = img_src, train_gt_src

        for i in range(args.re_ratio - 1): 
            img_src_con = np.concatenate((img_src_con, img_src))
            train_gt_src_con = np.concatenate((train_gt_src_con, train_gt_src))

        hyperparams_train = hyperparams.copy()
        hyperparams_train.update({'flip_augmentation': True, 'radiation_augmentation': True, 'mixture_augmentation': False})

        train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
        g = torch.Generator()
        g.manual_seed(args.seed)

        train_loader = data.DataLoader(train_dataset, batch_size=hyperparams['batch_size'], pin_memory=True,
                                   worker_init_fn=seed_worker, generator=g, shuffle=True)
        test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
        test_loader = data.DataLoader(test_dataset, pin_memory=True, # worker_init_fn=seed_worker, # generator=g,
                                  batch_size=hyperparams['batch_size'])
        
        all_dataset = HyperX(img_tar, gt_src, **hyperparams)
        all_loaders = data.DataLoader(all_dataset, pin_memory=True, batch_size=hyperparams['batch_size'])

        len_src_loader = len(train_loader)
        len_src_dataset = len(train_loader.dataset)
        len_tar_dataset = len(test_loader.dataset)
        len_tar_loader = len(test_loader)
        print(hyperparams)
        print("train samples:", len_src_dataset)
        print("test samples:", len_tar_dataset)

        correct, acc = 0, 0
        # pretrained_dict  = torch.jit.load('./ViT-B-32.pt', map_location="cpu").state_dict()
        # embed_dim = pretrained_dict["text_projection"].shape[1]   # 512
        # # print(embed_dim)
        # context_length = pretrained_dict["positional_embedding"].shape[0]   # 77
        # vocab_size = pretrained_dict["token_embedding.weight"].shape[0]   # 49408
        # transformer_width = pretrained_dict["ln_final.weight"].shape[0]   # 512
        # # transformer_heads = transformer_width // 64
        # # transformer_layers = 3
        # transformer_heads = transformer_width // 128   # 4
        # transformer_layers = 1   # 1

        embed_dim, context_length, vocab_size, transformer_width, transformer_heads, transformer_layers = 512, 77, 49408, 512, 4, 1
        model = FVMGN(embed_dim, img_src.shape[-1], hyperparams['patch_size'], gt_src.max(), context_length, 
                   vocab_size, transformer_width, transformer_heads, transformer_layers, hyperparams['source_name']).to(DEVICE)
    
        # for key in ["input_resolution", "context_length", "vocab_size"]:
        #     if key in pretrained_dict:
        #         del pretrained_dict[key]
    
        # model_dict = model.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'visual' not in k.split('.')}
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)

        # # =====================Get FLOPs and Params=====================
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params_model = total_trainable_params / (1024 * 1024)
        print(f'{total_trainable_params / (1024 * 1024):.4f}M training parameters.')
        inp_1 = torch.randn(BATCHSIZE, img_src.shape[2], PATCHSIZE, PATCHSIZE).to(DEVICE)
        inp_2 = torch.randn(BATCHSIZE, 77).to(DEVICE)
        inp_3 = torch.randn(BATCHSIZE).to(DEVICE)
        inp_4 = torch.randn(BATCHSIZE, 77).to(DEVICE)
        inp_5 = torch.randn(BATCHSIZE, 77).to(DEVICE)
        flops, paramss = profile(model, inputs=(inp_1, inp_2, inp_3, inp_4, inp_5))
        flops, paramss = clever_format([flops, paramss], "%.4f")
        print("FLOPs: ", flops)
        print("Params: ", paramss)
        # # ==============================================================
        # now_time = datetime.now()
        # time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
        # log_dir = os.path.join(args.save_path, time_str+'_lr_'+str(args.lr)+'_lam1_'+str(args.lambda_1)+'_alpha_'+str(args.alpha))
        # if not os.path.exists(log_dir):
        #     os.makedirs(log_dir)

        each_train = []
        each_test = []
        for epoch in range(1, args.num_epoch + 1): 
            t1_train = time.time()
            model, CCN_train_acc, label_tra, hsi_tra, oth_tra = train(epoch, model, args.num_epoch, LABEL_VALUES_src, LABEL_QUEUE)
            t2_train = time.time()
            each_time_train = t2_train - t1_train
            print('epoch time:', each_time_train)
            each_train.append(each_time_train)

            t1_test = time.time()
            t_correct, CCN_test_acc, pred, label, fea_hsi, fea_other = test(model, LABEL_VALUES_src)
            t2_test = time.time()
            each_time_test = t2_test - t1_test
            print('test time:', each_time_test)
            each_test.append(each_time_test)

            if t_correct > correct: 
                correct = t_correct
                acc = CCN_test_acc
                results = metrics(np.concatenate(pred), np.concatenate(label), ignored_labels=hyperparams['ignored_labels'], n_classes=gt_src.max())
                print(classification_report(np.concatenate(pred), np.concatenate(label), target_names=LABEL_VALUES_tar))
            print('{} and {} max correct: {} max accuracy{: .2f}%\n'.format(args.source_name, args.target_name, correct, 100. * correct / len_tar_dataset))

            # io.savemat(os.path.join(args.save_path, 'results_' + args.source_name+'_' + f'{CCN_test_acc * 100 :.2f}' + '.mat'), 
            #        {'lr':args.lr, 'lambda_1': args.lambda_1, 'alpha': args.alpha, 'results': results})

            # ===============================Plt cls map===============================
            acc_plot = 100. * t_correct / len_tar_dataset
            cls_maps = tensor2img(pred, test_gt_tar)
            classification_map(cls_maps, test_gt_tar, results['Overall_Accuracy'], DATASET, MODEL, hyperparams['patch_size'], TTimes=1)
            # ===============================Plt cls map===============================

        results_all.append(results)
        train_time = sum(each_train)
        test_time = sum(each_test) / len(each_test)
        total_time = train_time + test_time
        results['TrainTimes'] = train_time
        results['TestTimes'] = test_time
        results['SumTimes'] = total_time

        # =====================================Plt cls map (No)=====================================
        # cls_maps = tensor2img(pred, test_gt_tar)
        # cls_map(cls_maps, test_gt_tar, results['Overall_Accuracy'], DATASET, MODEL, hyperparams['patch_size'], TTimes=1)
        # =====================================================================================
        show_results(results, label_values=LABEL_VALUES_src, agregated=False, runs=N_RUNS)
        # write_results(hyperparams, results_all, params_model, label_values=LABEL_VALUES_src, agregated=True, runs=N_RUNS)

    if N_RUNS > 1: 
        show_results(results_all, label_values=LABEL_VALUES_src, agregated=True, runs=N_RUNS)
    
    write_results(hyperparams, results_all, params_model, flops, paramss, 
                  label_values=LABEL_VALUES_src, agregated=True, runs=N_RUNS, top_k=TOP_K)
