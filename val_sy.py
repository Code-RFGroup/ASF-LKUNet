import pandas as pd
import scipy.io as scio
import math
import scipy.io as sc
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils.metrics_sy import Train_index_sy
from torch.utils.data import Dataset, DataLoader
from utils.dataset_synapse_overlap import *
import glob
import os
import time
import argparse
import sys
from torchvision import transforms
from scipy.ndimage import zoom
import SimpleITK as sitk
from utils.overlap import *
from monai.inferers import sliding_window_inference

def val(args,model,logger, epoch,checkpoint,main_data_path,\
                    save_path,max_dice,weight_patch,device,save_pred=None):
    torch.cuda.empty_cache()
    logger.info('\n')
    logger.info('================testing================')
    test_name_list = open(os.path.join(main_data_path, 'Sy_test.txt')).readlines()

    #======================================parser args==========================================
    num_classes = args.num_classes
    dataset_type = args.dataset_type
    image_size = args.image_size
    in_chan = args.patch_size
    data_type = 'crop_'+str(image_size)+'_patch_'+str(in_chan)
    #======================================input path==========================================
    #input path
    test_path = main_data_path +'test_data/' + data_type + '/'
    logger.info('data input patch:{}'.format(test_path))


    #======================================label and indicator==========================================
    indicator_list=['Dice','ACC','Iou','F_score','Precision','Recall']
    label_list=["Aorta","Gallbladder","Kidney(L)","Kidney(R)","Liver","Pancreas","Spleen","Stomach","Avge"]  #avg
    # print(len(label_list))
    #======================================testing==========================================
    best_epoch = 0
    test_len = len(test_name_list)
    metric_list = 0.0
    model.eval()
    for i in range(test_len):
        # ======================================load data==========================================
        test_name = test_name_list[i].strip('\n')
        test_path_in = test_path + test_name + '/'
        test_set = load_data(test_path_in, image_size,'test',
                             transform=transforms.Compose(
                                 [RandomGenerator()]))
        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                  batch_size=1,
                                                  shuffle=False, num_workers=8)
        data_label = []
        data_pred = []
        with torch.no_grad():
            for image_input, label_input in tqdm(test_loader):      #single case slice
                image_input, label_input = image_input.float().to(device), label_input.float().to(device)
                output = sliding_window_inference(image_input,(image_size,image_size),4,model,overlap=0.75)
                output_ = torch.argmax(torch.softmax(output, dim=1), dim=1) #1,H,W
                # print(output.shape)
                data_label.append(label_input)     #raw_H,raw_W
                data_pred.append(output_)    #1,raw_H,raw_W
        # =================single case===============
        pred = torch.stack(data_pred).squeeze(1)    #slice,raw_H,raw_W
        mask = torch.stack(data_label).squeeze(1)    #slice,,raw_H,raw_W
        pred = pred.cpu().detach().numpy()
        mask = mask.cpu().detach().numpy()
        print(pred.shape,mask.shape)
        if save_pred is not None:
            prd_itk = sitk.GetImageFromArray(pred.astype(np.float32))
            sitk.WriteImage(prd_itk, save_path + '/' + test_name + "_pred.nii")

        #=================single indicators===============
        metric_ = Train_index_sy(pred, mask, num_classes,'train')    #single case indicators
        logger.info('case %s mean_dice %f ' % (test_name, np.mean(metric_, axis=0)[0]))
        metric_list += np.array(metric_)  # count all case indicators

    metric_list = metric_list / test_len  #mean all case indicators
    # print(len(metric_list))
    # =================mean all case indicators===============
    logger.info('total %d case mean'%(test_len))
    logger.info('\t\tmean_dice\tmacc\t\tmIou\t\tmF_score\tmPrecision\tmRecall ' )
    for i in range(1,num_classes):
        ii = i-1
        logger.info('%s\t%f\t%f\t%f\t%f\t%f\t%f' %
                                    (label_list[ii][:4], metric_list[ii][0],
                                     metric_list[ii][1], metric_list[ii][2],
                                     metric_list[ii][3], metric_list[ii][4],
                                     metric_list[ii][5]))
    Index = np.mean(metric_list, axis=0)
    logger.info('%s\t%f\t%f\t%f\t%f\t%f\t%f' %
                (label_list[-1], round(Index[0], 6),
                    round(Index[1], 6), round(Index[2], 6),
                    round(Index[3], 6), round(Index[4], 6),
                    round(Index[5], 6)))
    best_metric_list = []
    # ================================save best dice model====================================
    logger.info('max_dice %f ' %(round(max_dice,6)))
    if max_dice < Index[0]:
        max_dice = Index[0]
        best_metric_list = metric_list.tolist()
        best_metric_list.append(Index.tolist())
        best_epoch = epoch
        best_metric_list = np.array(best_metric_list)
        best_metric_list = np.round(best_metric_list,6)
        logger.info('best dice epoch {}'.format(epoch))
        # logger.info('best dice hd95 {}  {}'.format(Index[0], Index[1]))
        logger.info('best dice {}'.format(Index[0].round(6)))

        if not os.path.exists(weight_patch):
            os.makedirs(weight_patch)
        torch.save(checkpoint, weight_patch + 'best_'+str(best_epoch)+'.pth')
        logger.info('save model, best epoch {} '.format(epoch))

        # print(best_metric_list)
        save_csv = pd.DataFrame(best_metric_list, columns=indicator_list)
        save_csv.insert(0, 'Class', label_list)
        save_csv.insert(len(indicator_list)+1, 'Best_epoch', best_epoch)
        save_csv.to_csv(save_path+dataset_type+'.csv',index=False,sep=',')
    print('\n')
    return max_dice






