# -*- coding: utf-8 -*-
import random
import numpy as np
import os
import re
import h5py
import torch
import imageio
import datetime
import spectral
import itertools
import numpy as np
from scipy import io
import matplotlib as mpl
import sklearn.model_selection
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import cv2 as cv
from PIL import Image, ImageEnhance


# ================================================================================================================
def pad_img(X, patches):
    height, width, band = X.shape[0], X.shape[1], X.shape[2]
    if patches % 2 == 0:
        padded_data = np.zeros((height + int(patches), width + int(patches), band))
    elif patches % 2 != 0:
        padded_data = np.zeros((height + int(patches-1), width + int(patches-1), band))
    for i in range(band):
        if patches % 2 == 0:
            padded_data[:,:,i] = np.pad(X[:,:,i], int((patches)/2), 'symmetric')
        elif patches % 2 != 0:
            padded_data[:,:,i] = np.pad(X[:,:,i], int((patches-1)/2), 'symmetric')
    return padded_data
# ================================================================================================================
def pad_gt(X, patches):
    height, width = X.shape[0], X.shape[1]
    if patches % 2 != 0:
        padded_data = np.zeros((height + int(patches-1), width + int(patches-1)))
        padded_data[:,:] = np.pad(X[:,:], int((patches-1)/2), 'symmetric')
    elif patches % 2 == 0:
        padded_data = np.zeros((height + int(patches), width + int(patches)))
        padded_data[:,:] = np.pad(X[:,:], int((patches)/2), 'symmetric')
    return padded_data
# ================================================================================================================

def get_device(ordinal):
    # Use GPU ?
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device


def seed_worker(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_deterministic(seed):
    print('Deterministic mode, seed:', seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        return io.loadmat(dataset)
        # return h5py.File(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return imageio.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))


def sliding_window(image, step=10, window_size=(20, 20), with_data=True):
    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    for x in range(0, W - w + offset_w, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h


def metrics(prediction, target, ignored_labels=[], n_classes=None):
    ignored_mask = np.zeros(target.shape[:2], dtype=np.bool_)
    for l in ignored_labels: 
        ignored_mask[target == l] = True
    ignored_mask = ~ignored_mask
    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes
    n_classes = int(n_classes)
    cm = confusion_matrix(target, prediction, labels=range(n_classes))

    results["Confusion_Matrix"] = cm

    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    results["TPR"] = TPR

    # Compute overall accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    results["Overall_Accuracy"] = accuracy

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2 * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1
    results["F1_Scores"] = F1scores

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa

    # Compute AA score
    shape = np.shape(cm)
    number = 0
    sums = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]): 
        number += cm[i, i]
        AA[i] = cm[i, i] / np.sum(cm[i, :])
        sums += np.sum(cm[i, :]) * np.sum(cm[:, i])
    AA_mean = np.mean(AA * 100)

    results['Class_Accuracy'] = AA
    results['Average_Accuracy'] = AA_mean
    results["prediction"] = prediction
    results["label"] = target

    return results


def show_results(results, label_values=None, agregated=False, runs=1):
    text = ""
    if agregated:
        cm = np.mean([r["Confusion_Matrix"] for r in results], axis=0)
        F1_scores = [r["F1_Scores"] for r in results]
        F1_scores_mean = np.mean(F1_scores, axis=0)
        F1_scores_std = np.std(F1_scores, axis=0)
        class_accuracies = [r['Class_Accuracy'] for r in results]
        Class_mean = np.mean(class_accuracies, axis=0)
        Class_std = np.std(class_accuracies, axis=0)
        accuracies = [r["Overall_Accuracy"] for r in results]
        aa_accuracies = [r['Average_Accuracy'] for r in results]
        kappas = [r["Kappa"] for r in results]
        text += "Agregated results: \n"
    else:
        cm = results["Confusion_Matrix"]
        F1scores = results["F1_Scores"]
        CAs = results['Class_Accuracy']
        AA = results['Average_Accuracy']
        OA = results["Overall_Accuracy"]
        kappa = results["Kappa"]

    text += "Confusion_Matrix: \n"
    text += str(cm)
    text += "\n----------------------------------------------------------------\n"

    text += "F1_Scores: \n"
    if agregated:
        for label, score, std in zip(label_values, F1_scores_mean, F1_scores_std):
            text += "\t{}: {:.04f} ± {:.04f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, F1scores):
            text += "\t{}: {:.04f}\n".format(label, score)
    text += "----------------------------------------------------------------\n"

    text += "CA:\n"
    acc_class=[]
    if agregated:
        for label_c, acc_c, std_c in zip(label_values, Class_mean, Class_std):
            text += "\t{}: {:.02f} ± {:.02f}\n".format(label_c, acc_c*100, std_c*100)
            acc_class.append(acc_c)
        acc_classes = [float('{:.04f}'.format(i)) for i in acc_class]
        text += ('CAs: ' + str(acc_classes))
    else:
        for label_c, acc_c in zip(label_values, CAs):
            text += "\t{}: {:.04f}\n".format(label_c, acc_c)
            acc_class.append(acc_c)
        acc_classes = [float('{:.04f}'.format(i)) for i in acc_class]
        text+=('CAs: ' + str(acc_classes))
    text += "\n----------------------------------------------------------------\n"

    if agregated: 
        text += ("OA: {:.02f} ± {:.02f}\n".format(np.mean(accuracies), np.std(accuracies)))
    else:
        text += "OA: {:.02f}\n".format(OA)

    if agregated:
        text += ("AA: {:.02f} ± {:.02f}\n".format(np.mean(aa_accuracies), np.std(aa_accuracies)))
    else:
        text += "AA: {:.02f}\n".format(AA)

    if agregated: 
        text += ("Kappa: {:.02f} ± {:.02f}\n".format(np.mean(kappas*100), np.std(kappas*100)))
    else:
        text += "Kappa: {:.02f}\n".format(kappa*100)
    text += "----------------------------------------------------------------\n"

    if agregated:
        if runs > 1: 
            AA_all = []
            OA_all = []
            Kappa_all = []
            TrainTime=[]
            TestTime=[]
            SumTime=[]
            for i in range(0, runs):
                AA_all.append(results[i]['Average_Accuracy']/100)
                OA_all.append(results[i]['Overall_Accuracy']/100)
                TrainTime.append(results[i]['TrainTimes'])
                TestTime.append(results[i]['TestTimes'])
                SumTime.append(results[i]['SumTimes'])
                Kappa_all.append(results[i]['Kappa'])
            AA_all_mean = np.mean(AA_all, axis=0)
            OA_all_mean = np.mean(OA_all, axis=0)
            Kappa_all_mean=np.mean(Kappa_all, axis=0)

            AA_all = [float('{:.04f}'.format(i)) for i in AA_all]
            OA_all = [float('{:.04f}'.format(i)) for i in OA_all]
            Kappa_all = [float('{:.04f}'.format(i)) for i in Kappa_all]
            
            text += "================================================================\n"
            text +=('The Number of Runs: {}\n'.format(runs))
            text += "----------------------------------------------------------------\n"
            text+=('OA_all: '+str(OA_all)+'\n')
            text += "OA_all_mean : {:.04f}\n".format(OA_all_mean*100)
            text += "----------------------------------------------------------------\n"
            text+=('AA_all: '+str(AA_all)+'\n')
            text += "AA_all_mean : {:.04f}\n".format(AA_all_mean*100)
            text += "----------------------------------------------------------------\n"
            text+=('Kappa_all: ' + str(Kappa_all)+'\n')
            text += "Kappa_all_mean : {:.04f}\n".format(Kappa_all_mean*100)
            text += "----------------------------------------------------------------\n"
            text+='TrainTimes: '+str(TrainTime)+'\n'
            text+='TrainTime_mean: {:.4f}\n'.format(np.mean(TrainTime))
            text += "----------------------------------------------------------------\n"
            text+='TestTimes: '+str(TestTime)+'\n'
            text+='TestTime_mean: {:.4f}\n'.format(np.mean(TestTime))
            text += "----------------------------------------------------------------\n"
            text+='SumTimes: '+str(SumTime)+'\n'
            text+='SumTime_mean: {:.4f}\n'.format(np.mean(SumTime))
            text += "================================================================\n"
    # vis.text(text.replace('\n', '<br/>'))
    print(text)


def write_results(hyperparams_modify, results, params_model, flops, paramss, label_values=None, agregated=False, runs=1, top_k=1):
    """Optional: This will overwrite the last result."""
    # *********************Results*************************
    # f=open('./results.txt', "w+",encoding='utf-8')
    # f.truncate()
    # f.close()
    # ***************************************************
    with open ('./Res.txt', 'a+', encoding='utf-8') as fp:
        fp.write('*******{}---{}*******\n'.format(hyperparams_modify['model'], datetime.datetime.strftime(datetime.datetime.now(),'%Y.%m.%d-%H:%M:%S')))
        fp.write('\n')
        fp.write('Model: {}\n'.format(hyperparams_modify['model']))
        fp.write('Source Dataset: {}\n'.format(hyperparams_modify['source_name']))
        fp.write('Target Dataset: {}\n'.format(hyperparams_modify['target_name']))
        fp.write('Training Sample: {}\n'.format(hyperparams_modify['training_sample_ratio']))
        if hyperparams_modify['lr'] != None:
            fp.write('Learning Rate: {}\n'.format(hyperparams_modify['lr']))
        else:
            pass
        fp.write('Batch Size: {}\n'.format(hyperparams_modify['batch_size']))
        fp.write('Patch Size: {}\n'.format(hyperparams_modify['patch_size']))
        fp.write('Epoch: {}\n'.format(hyperparams_modify['num_epoch']))
        fp.write('\n')
        if agregated:
            F1_scores = [r["F1_Scores"] for r in results]
            class_accuracies = [r['Class_Accuracy'] for r in results]
            global_accuracies = [r["Overall_Accuracy"] for r in results]
            aa_accuracies = [r['Average_Accuracy'] for r in results]
            kappas = [r["Kappa"] for r in results]
            F1_scores_mean = np.mean(F1_scores, axis=0)
            F1_scores_std = np.std(F1_scores, axis=0)
            Class_mean = np.mean(class_accuracies, axis=0)
            Class_std = np.std(class_accuracies, axis=0)
            cm = np.mean([r["Confusion_Matrix"] for r in results], axis=0)
            fp.write("******Agregated Results******\n")
        else:
            cm = results["Confusion matrix"]
            global_accuracy = results["Overall_Accuracy"]
            AAs=results['Class_Accuracy']
            AA_means=results['Average_Accuracy']
            F1scores = results["F1_Scores"]
            kappa = results["Kappa"]
        fp.write("----------------------------------------------------------------\n")
        # label_values = label_values[1:]

        # fp.write("Confusion matrix: \n")
        # fp.write(str(cm))
        # fp.write("\n----------------------------------------------------------------\n")

        fp.write("F1 Scores: \n")
        if agregated:
            for label, score, std in zip(label_values, F1_scores_mean, F1_scores_std):
                fp.write("\t{}: {:.04f} ± {:.04f}\n".format(label, score, std))
        else:
            for label, score in zip(label_values, F1scores):
                fp.write("\t{}: {:.04f}\n".format(label, score))
        fp.write("----------------------------------------------------------------\n")

        fp.write("Class Accuracy: \n")
        acc_class=[]
        if agregated:
            for label_c, acc_c, std_c in zip(label_values, Class_mean, Class_std):
                fp.write("\t{}: {:.02f} ± {:.02f}\n".format(label_c, acc_c*100, std_c*100))
                acc_class.append(acc_c)
            acc_classes = [float('{:.04f}'.format(i)) for i in acc_class]
            fp.write(('All Class Accuracy: '+str(class_accuracies)))
            fp.write("\n")
            fp.write(('Class Accuracy: '+str(acc_classes)))
        else: 
            for label_c, acc_c in zip(label_values, AAs):
                fp.write("\t{}: {:.04f}\n".format(label_c, acc_c))
                acc_class.append(acc_c)
            acc_classes = [float('{:.04f}'.format(i)) for i in acc_class]
            fp.write(('Class Accuracy: '+str(acc_classes)))
        fp.write("\n----------------------------------------------------------------\n")

        if agregated:
            fp.write(("Overall Accuracy: {:.04f} ± {:.04f}\n".format(np.mean(global_accuracies), np.std(global_accuracies))))
        else:
            fp.write("Overall Accuracy: {:.04f}\n".format(global_accuracy))
        fp.write("----------------------------------------------------------------\n")

        if agregated:
            fp.write(("Average Accuracy: {:.04f} ± {:.04f}\n".format(np.mean(aa_accuracies), np.std(aa_accuracies))))
        else:
            fp.write("Average Accuracy: {:.04f}\n".format(AA_means))
        fp.write("----------------------------------------------------------------\n")
        if agregated:
            fp.write(("Kappa: {:.02f} ± {:.02f}\n".format(np.mean(kappas)*100, np.std(kappas)*100)))
        else:
            fp.write("Kappa: {:.02f}\n".format(kappa*100))
            fp.write("================================================================\n")

        fp.write("================================================================\n")
        fp.write("Accuracy:\n")
        if agregated:
            for label_c, acc_c, std_c in zip(label_values, Class_mean, Class_std):
                fp.write("{:.02f} ± {:.02f}\n".format(acc_c*100, std_c*100))
        else:
            pass
        if agregated:
            fp.write(("{:.02f} ± {:.02f}\n".format(np.mean(global_accuracies), np.std(global_accuracies))))
        else:
            fp.write("{:.02f}\n".format(global_accuracy))
        if agregated:
            fp.write(("{:.02f} ± {:.02f}\n".format(np.mean(aa_accuracies), np.std(aa_accuracies))))
        else:
            fp.write("{:.02f}\n".format(AA_means))
        if agregated:
            fp.write(("{:.02f} ± {:.02f}\n".format(np.mean(kappas)*100, np.std(kappas)*100)))
        else:
            fp.write("{:.02f}\n".format(kappa*100))
            fp.write("================================================================\n")
    
        if agregated:
            if runs > 1:
                AA_all = []
                OA_all = []
                Kappa_all = []
                TrainTime=[]
                TestTime=[]
                SumTime=[]
                for i in range(0, runs): 
                    AA_all.append(results[i]['Average_Accuracy']/100)
                    OA_all.append(results[i]['Overall_Accuracy']/100)
                    TrainTime.append(results[i]['TrainTimes'])
                    TestTime.append(results[i]['TestTimes'])
                    SumTime.append(results[i]['SumTimes'])
                    Kappa_all.append(results[i]['Kappa'])
                AA_all_mean = np.mean(AA_all, axis=0)
                OA_all_mean = np.mean(OA_all, axis=0)
                Kappa_all_mean=np.mean(Kappa_all, axis=0)

                AA_all = [float('{:.04f}'.format(i)) for i in AA_all]
                OA_all = [float('{:.04f}'.format(i)) for i in OA_all]
                Kappa_all = [float('{:.04f}'.format(i)) for i in Kappa_all]

                indexed_OA_all = list(enumerate(OA_all))
                OA_top_10 = sorted(indexed_OA_all, key=lambda x: x[1], reverse=True)[:top_k]
                OA_k_indices, OA_k_values = zip(*OA_top_10)
                AA_k_values = [AA_all[i] for i in OA_k_indices]
                Kappa_k_values = [Kappa_all[i] for i in OA_k_indices]
                cls_k_values = [class_accuracies[i] for i in OA_k_indices]
                TrainTime_k = [TrainTime[i] for i in OA_k_indices]
                TestTime_k = [TestTime[i] for i in OA_k_indices]
                SumTime_k = [SumTime[i] for i in OA_k_indices]
            
                fp.write("================================================================\n")
                fp.write(('The Number of Runs: {}\n'.format(runs)))
                fp.write("----------------------------------------------------------------\n")
                fp.write("Samples: {}\n".format(hyperparams_modify['training_sample_ratio']))
                fp.write("----------------------------------------------------------------\n")
                fp.write("Training Params: {:.04f}\n".format(params_model))
                fp.write("----------------------------------------------------------------\n")
                fp.write("FLOPs: {}\n".format(flops))
                fp.write("----------------------------------------------------------------\n")
                fp.write("Params: {}\n".format(paramss))
                fp.write("----------------------------------------------------------------\n")
                fp.write(('OA_all: '+str(OA_all)+'\n'))
                fp.write("{} times OA_all_mean: {:.02f}\n".format(runs, OA_all_mean*100))
                fp.write("----------------------------------------------------------------\n")
                fp.write(('AA_all: '+str(AA_all)+'\n'))
                fp.write("{} times AA_all_mean: {:.02f}\n".format(runs, AA_all_mean*100))
                fp.write("----------------------------------------------------------------\n")
                fp.write(('Kappa_all: '+str(Kappa_all)+'\n'))
                fp.write("{} times Kappa_all_mean: {:.02f}\n".format(runs, Kappa_all_mean*100))
                fp.write("----------------------------------------------------------------\n")
                fp.write('TrainTimes: '+str(TrainTime)+'\n')
                fp.write('{} times TrainTime_mean: {:.4f}\n'.format(runs, np.mean(TrainTime)))
                fp.write("----------------------------------------------------------------\n")
                fp.write('TestTimes: '+str(TestTime)+'\n')
                fp.write('{} times TestTime_mean: {:.4f}\n'.format(runs, np.mean(TestTime)))
                fp.write("----------------------------------------------------------------\n")
                fp.write('SumTimes: '+str(SumTime)+'\n')
                fp.write('{} times SumTime_mean: {:.4f}\n'.format(runs, np.mean(SumTime)))
                fp.write("============================RESULTS==============================\n")
                fp.write("FINAL RESULTS:\n")
                for mean, std in zip(np.mean(cls_k_values, axis=0)*100, np.std(cls_k_values, axis=0)*100):  
                    fp.write("{:.02f} ± {:.02f}\n".format(mean, std))
                fp.write(("{:.02f} ± {:.02f}\n".format(np.mean(OA_k_values)*100, np.std(OA_k_values)*100)))
                fp.write(("{:.02f} ± {:.02f}\n".format(np.mean(AA_k_values)*100, np.std(AA_k_values)*100)))
                fp.write(("{:.02f} ± {:.02f}\n".format(np.mean(Kappa_k_values)*100, np.std(Kappa_k_values)*100)))
                fp.write("{}\n".format(flops))
                fp.write("{}\n".format(paramss))
                fp.write(("{:.02f}\n".format(np.mean(TrainTime_k))))
                fp.write(("{:.02f}\n".format(np.mean(TestTime_k))))
                fp.write(("{:.02f}\n".format(np.mean(SumTime_k))))
                fp.write("============================SUCCESS==============================\n")
                fp.write("\n")
                fp.write("\n")
                fp.write("\n")
            elif runs == 1: 
                AA_all = []
                OA_all = []
                Kappa_all = []
                TrainTime=[]
                TestTime=[]
                SumTime=[]
                for i in range(runs): 
                    AA_all.append(results[i]['Average_Accuracy']/100)
                    OA_all.append(results[i]['Overall_Accuracy']/100)
                    TrainTime.append(results[i]['TrainTimes'])
                    TestTime.append(results[i]['TestTimes'])
                    SumTime.append(results[i]['SumTimes'])
                    Kappa_all.append(results[i]['Kappa'])
                AA_all_mean = np.mean(AA_all, axis=0)
                OA_all_mean = np.mean(OA_all, axis=0)
                Kappa_all_mean=np.mean(Kappa_all, axis=0)

                AA_all = [float('{:.04f}'.format(i)) for i in AA_all]
                OA_all = [float('{:.04f}'.format(i)) for i in OA_all]
                Kappa_all = [float('{:.04f}'.format(i)) for i in Kappa_all]
            
                fp.write("================================================================\n")
                fp.write(('The Number of Runs: {}\n'.format(runs)))
                fp.write("Samples: {}\n".format(hyperparams_modify['training_sample_ratio']))
                fp.write("Training Params: {:.04f}\n".format(params_model))
                fp.write("FLOPs: {}\n".format(flops))
                fp.write("Params: {}\n".format(paramss))
                fp.write('TrainTimes: '+str(TrainTime)+'\n')
                fp.write('TestTimes: '+str(TestTime)+'\n')
                fp.write('SumTimes: '+str(SumTime)+'\n')
                fp.write("============================SUCCESS==============================\n")
                fp.write("\n")
                fp.write("\n")
                fp.write("\n")


def sample_gt(gt, train_size, mode='random'):
    indices = np.nonzero(gt)
    X = list(zip(*indices)) # x, y features
    y = gt[indices].ravel() # classes
    num_class = int(y.max())
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
       train_size = int(train_size)
    train_label = []
    test_label = []
    if mode == 'random': 
        if train_size == 1:
            random.shuffle(X)
            train_indices = [list(t) for t in zip(*X)]
            [train_label.append(i) for i in gt[tuple(train_indices)]]
            train_set = np.column_stack((train_indices[0], train_indices[1], train_label))
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt = []
            test_set = []
        else:
            train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y, random_state=23)
            train_indices = [list(t) for t in zip(*train_indices)]
            test_indices = [list(t) for t in zip(*test_indices)]
            train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
            test_gt[tuple(test_indices)] = gt[tuple(test_indices)]

            [train_label.append(i) for i in gt[tuple(train_indices)]]
            train_set = np.column_stack((train_indices[0], train_indices[1], train_label))
            [test_label.append(i) for i in gt[tuple(test_indices)]]
            test_set = np.column_stack((test_indices[0], test_indices[1], test_label))

    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / second_half_count
                    if ratio > 0.9 * train_size and ratio < 1.1 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0
        test_gt[train_gt > 0] = 0
    
    elif mode == 'fixed_num':
        train_gt_in = []
        test_gt_in = []
        all_train_samples = 0
        for class_label in range(1, num_class + 1):
            class_indices = np.where(gt == class_label)
            class_indices = np.array(class_indices).T
            arr_class = np.arange(0, class_indices.shape[0])
            if class_indices.shape[0] <= train_size:
                num_train_samples = class_indices.shape[0] // 2
            else:
                num_train_samples = train_size
            all_train_samples += num_train_samples
 
            train_indices = np.random.choice(class_indices.shape[0], size=num_train_samples, replace=False)
            train_samples = class_indices[train_indices]
            train_gt_in.extend(train_samples)

            test_indices = np.setdiff1d(arr_class, train_indices)
            test_samples = class_indices[test_indices]
            test_gt_in.extend(test_samples)

        train_gt_in = np.array(train_gt_in).reshape((all_train_samples, 2))
        test_gt_in = np.array(test_gt_in).reshape(((len(y) - all_train_samples), 2))
        train_gt_in = [(x, y) for x, y in train_gt_in]
        test_gt_in = [(x, y) for x, y in test_gt_in]

        train_gt_in = [list(t) for t in zip(*train_gt_in)]
        test_gt_in = [list(t) for t in zip(*test_gt_in)]
        train_gt[tuple(train_gt_in)] = gt[tuple(train_gt_in)]
        test_gt[tuple(test_gt_in)] = gt[tuple(test_gt_in)]

        [train_label.append(i) for i in gt[tuple(train_gt_in)]]
        train_set = np.column_stack((train_gt_in[0], train_gt_in[1], train_label))
        [test_label.append(i) for i in gt[tuple(test_gt_in)]]
        test_set = np.column_stack((test_gt_in[0], test_gt_in[1], test_label))

    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt, train_set, test_set


def tensor2img(pred, gt_tar):
    new_show = np.zeros((gt_tar.shape[0], gt_tar.shape[1]))
    k = 0
    predict_label = np.concatenate(pred).flatten()
    for i in range(gt_tar.shape[0]):
        for j in range(gt_tar.shape[1]):
            if gt_tar[i][j] != 0:
                new_show[i][j] = predict_label[k]
                new_show[i][j] += 1
                k += 1
    return new_show

def classification_map(seg_map, ground_truth, oa, data_set, model, patch_size, TTimes=1):
    cmap = np.asarray([[0, 0, 255], [255, 100, 0], [0, 255, 134], [150, 70, 150], [100, 150, 255], [60, 90, 114],
                  [255, 255, 125], [255, 0, 255], [100, 0, 255], [1, 170, 255], [0, 255, 0], [175, 175, 82],
                  [100, 190, 56], [140, 67, 46], [115, 255, 172], [255, 255, 0], [75, 125, 80], [125, 125, 0], 
                  [35, 75, 40], [75, 75, 0]]) / 255.
    background = np.array([255, 255, 255]) / 255.
    x_re = seg_map.flatten()
    y = np.zeros((x_re.shape[0], 3))
    for i in range(len(np.unique(ground_truth)) - 1):
        y[x_re == i + 1] = cmap[i]
    y[x_re == 0] = background
    y_re = np.reshape(y, (ground_truth.shape[0], ground_truth.shape[1], 3))
    first = patch_size // 2
    end_1 = ground_truth.shape[0] - first
    end_2 = ground_truth.shape[1] - first
    y_re = y_re[first:end_1, first:end_2]
    print(y_re.shape)

    fig = plt.figure(frameon=False)
    ax=plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(y_re)
    save_path = './Figure/{}_{}_{:.2f}%_{:.2f}s_{}.pdf'.format(data_set, model, oa, TTimes, datetime.datetime.strftime(datetime.datetime.now(),'%Y.%m.%d_%H.%M.%S'))
    fig.savefig(save_path, bbox_inches='tight', pad_inches = 0, transparent=True)
    plt.close()


def hisEqulColor(img, alpha=0.2): 
    img_yuv = cv.cvtColor(img, cv.COLOR_RGB2YUV)
    y_channel = img_yuv[:, :, 0]
    y_eq = cv.equalizeHist(y_channel)
    img_yuv[:, :, 0] = cv.addWeighted(y_channel, 1 - alpha, y_eq, alpha, 0)
    img_output = cv.cvtColor(img_yuv, cv.COLOR_YUV2RGB)
    return img_output


def save_figs(Data, Label, Datasetname1, Datasetname2, Channels=(40, 27, 13)):  # 
    num_classes = len(np.unique(Label))
    sample_each_class = np.asarray([Label[Label == k] for k in range(1, num_classes)], dtype=object)
    num_each_class = [len(sample_each_class[k]) for k in range(num_classes - 1)]
    SumLable, SumAll = np.sum(num_each_class), Label.shape[0] * Label.shape[1]
    print('Data Shape:', Data.shape)
    print('Ground Truth Shape:', Label.shape)
    print('Num CLS:', num_classes - 1)
    print('Num Each CLS: ', num_each_class)
    print('Num Labeled Samples {}, Num All Samples {}, Num Unlabeled Samples {}, Percent Labeled Samples {:.2f}%'.format(SumLable, SumAll, SumAll - SumLable, SumLable / SumAll * 100))

    data1 = np.zeros((Data.shape[0], Data.shape[1], 3))
    data1[:, :, 0], data1[:, :, 1], data1[:, :, 2] = Data[:, :, Channels[0]], Data[:, :, Channels[1]], Data[:, :, Channels[2]]
    for b in range(3): 
        temp = data1[:, :, b]
        data1[:, :, b] = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
    data1 = hisEqulColor((data1 * 255.).astype(np.uint8))

    cmap = np.asarray([[0, 0, 255], [255, 100, 0], [0, 255, 134], [150, 70, 150], [100, 150, 255], [60, 90, 114],
                  [255, 255, 125], [255, 0, 255], [100, 0, 255], [1, 170, 255], [0, 255, 0], [175, 175, 82],
                  [100, 190, 56], [140, 67, 46], [115, 255, 172], [255, 255, 0], [75, 125, 80], [125, 125, 0], 
                  [35, 75, 40], [75, 75, 0]]) / 255.
    background = np.array([255, 255, 255]) / 255.

    x_re = Label.flatten()
    y = np.zeros((x_re.shape[0], 3))
    for i in range(len(np.unique(Label)) - 1):
        y[x_re == i + 1] = cmap[i]
    y[x_re == 0] = background
    y_re = np.reshape(y, (Label.shape[0], Label.shape[1], 3))
    
    fig = plt.figure(frameon=False)
    ax=plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(y_re)
    save_path = './Original/{}_gt.pdf'.format(Datasetname1)
    fig.savefig(save_path, bbox_inches='tight', pad_inches = 0, transparent=True)
    save_path1 = './Original/{}_gt.png'.format(Datasetname1)
    fig.savefig(save_path1, bbox_inches='tight', pad_inches=0, transparent=True, dpi=300)
    plt.close()

    fig1 = plt.figure(frameon=False)
    ax1=plt.Axes(fig1, [0., 0., 1., 1.])
    ax1.set_axis_off()
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    fig1.add_axes(ax1)
    ax1.imshow(data1, cmap='gray')
    save_path = './Original/{}_ori.pdf'.format(Datasetname2)
    fig1.savefig(save_path, bbox_inches='tight', pad_inches = 0, transparent=True)
    save_path1 = './Original/{}_ori.png'.format(Datasetname2)
    fig1.savefig(save_path1, bbox_inches='tight', pad_inches=0, transparent=True, dpi=300)
    plt.close()
