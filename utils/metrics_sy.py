from heapq import merge
import numpy as np
import torch
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from medpy import metric
import time

class DiceLoss_sy(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss_sy, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        # class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            # class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        # dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return  hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 0
    else:
        return 0

def Train_index(pred, gt,Type = 'train'):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    #x,y,z =  y_ture.shape
    TP = np.sum(gt * pred)
    FP = np.sum(pred * (1 - gt))
    FN = np.sum((1 - pred) * (gt))
    TN = np.sum((1 - gt) * (1 - pred))

    #print(TP,FP,FN)
    #Recall not nan
    if (TP + FN) == 0:
        Recall = 0
    else:
        Recall = TP / (TP + FN)
    #Precision not nan
    if (TP + FP) == 0:
        Precision = 0
    else:
        Precision = TP / (TP + FP)
    #F_score not nan
    if (Recall + Precision) == 0:
        F_score = 0
    else:
        F_score = (2 * Recall * Precision) / (Recall + Precision)
    #ACC not nan
    if (TP + FP + FN ) ==0:
        Iou = 0
    else:
        Iou = TP / (TP + FP + FN )

    ACC = (TP+TN) /(TP+TN+FP+FN)
    if Recall is None:
        Recall = 0
    elif Precision is None:
        Precision = 0
    elif F_score is None:
        F_score = 0

    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        if Type == 'test':
            hd95 = metric.binary.hd95(pred, gt)
    elif pred.sum() > 0 and gt.sum()==0:
        dice =  0
        hd95 = 0
    else:
        dice =  0
        hd95 = 0
    if Type == 'train':
        return round(dice,6), round(ACC,6), round(Iou,6), \
            round(F_score,6), round(Precision,6), round(Recall,6)
    elif Type == 'test':
        return round(dice,6), round(hd95,6),round(ACC,6), round(Iou,6), \
            round(F_score,6), round(Precision,6), round(Recall,6)


def Train_index_sy(image, label, classes,Type = 'train'):
    Type = Type
    metric_list = []
    start = time.time()
    for i in range(1, classes):
        metric_list.append(Train_index(image == i, label == i,Type = Type))
    end = time.time()
    print('calculating time(s): ',end-start)
    # print(len(metric_list))
    return metric_list