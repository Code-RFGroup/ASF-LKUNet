from itertools import zip_longest
import torch
import argparse
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
from utils.metrics_sy import *
from torch.nn.modules.loss import CrossEntropyLoss
from utils.dataset_acdc import *
from models.model import ASF_LKUNet
from tqdm import tqdm
import time
from matplotlib import pyplot as plt
import time
from utils.getlogger import get_logger
import sys
from torchvision import transforms
from thop import profile
from val_acdc import val
import pandas as pd



torch.cuda.empty_cache()
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    default='', help='root dir for data')
parser.add_argument('--root_path', type=str,
                    default='', help='root dir for data')
parser.add_argument('-j', '--num_workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run(default: 400)')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N', help='batch size (default: 1)')
parser.add_argument('--learning_rate', default=0.00005, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.99, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--learning_decay', type=int, default=5)
parser.add_argument('--loss_type', type=str, default='CE+Dice_config_ablation_conv_0_SE_5_')
parser.add_argument('--break_point', type=str, default=' ')
parser.add_argument('--dataset_type', type=str, default='ACDC')
parser.add_argument('--patch_size', type=int, default=0)
parser.add_argument('--num_classes', type=int, default=4)
parser.add_argument('--load_model_num', type=int, default=15)
parser.add_argument('--save_model_epoch', type=int, default=50)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--val', type=str, default='True')
args = parser.parse_args()


# seed = 3000
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)

Time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
Time = Time.replace('-','').replace(' ','_').replace(':','')

#================================path===============================
main_data_path = args.data_path
main_path = args.root_path

#================================ parser args===============================
dataset_type = args.dataset_type
num_classes = args.num_classes
patch_size = args.patch_size
image_size = args.image_size
start_epoch = args.load_model_num
window_size = (args.window_size,args.window_size)
print('patch size',patch_size)
print('num_classes',num_classes)
if patch_size==0:
    print('error patch size!')
    import sys
    sys.exit(0)
if args.break_point=='True' :
   break_point = 1
#    start_epoch= 0
elif args.break_point=='False':
    break_point = 0
    start_epoch= 0
else:
    print('error break_point type!')
    sys.exit(0)

if args.train_in_test=='True' :
    train_in_test = 1
elif args.train_in_test=='False':
    train_in_test = 0
else:
    print('error train_in_test type!')
    sys.exit(0)

in_chan = patch_size
data_type = 'crop_'+str(image_size)+'_patch_'+str(in_chan)
max_dice = 0
#================================ input path ====================================
loss_type  = args.loss_type + args.re_type
train_path = main_data_path +'train_data_'+args.re_type[:-2]+'/'+ data_type+'/'
save_path = main_path + 'test_result/'+ dataset_type+'/'+data_type +'_'+loss_type+'/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

record_type = data_type + '_'+loss_type      #save train loss logger ==> jpg
data_type = dataset_type+str(in_chan)
weight_patch = main_path +'/work_dirs/weight/'+data_type+'_save_each_eopch_'+loss_type+'/'
if not os.path.exists(weight_patch):
    os.makedirs(weight_patch)
if not os.path.exists(weight_patch.replace('weight','logger')):
    os.makedirs(weight_patch.replace('weight','logger'))
logger = get_logger(weight_patch.replace('weight','logger')+str(Time)+'.log',
                    verbosity=1, name=__name__)

#================================model per====================================
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda")
model = ASF_LKUNet(n_channels=in_chan, n_classes=num_classes).to(device)
logger.info('model {}'.format(model))
model.to(device)

##=========configs===========##
logger.info('Data Type {}'.format(data_type))
logger.info('max epoch {}'.format(args.epochs))
logger.info('batch size {}'.format(args.batch_size))
logger.info('learning rate {}'.format(args.learning_rate))
logger.info('learning decay {}'.format(args.learning_decay))
logger.info('loss type {}'.format(loss_type))
logger.info('patch size {}'.format(in_chan))
logger.info('dataset type {}'.format(dataset_type))
logger.info('num_classes {}'.format(num_classes))
logger.info('save_model_epoch {}'.format(args.save_model_epoch))
logger.info('image_size {}'.format(image_size))
logger.info('break_point {}'.format(args.break_point))

#================================load data====================================
logger.info('train path {}'.format(train_path))
train_dataset = load_data(train_path,image_size,'train',
                            transform=transforms.Compose(
                                        [RandomGenerator()]))
# trianloader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True,
#                             num_workers=args.num_workers,drop_last=True)
trianloader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True,
                            num_workers=args.num_workers)
    
#================================model optimizer====================================
criterion_ce = CrossEntropyLoss()
criterion_dice = DiceLoss_sy(num_classes)

optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate,
                             weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.learning_decay, gamma=0.99)

log_dict = {'celoss': [],'diceloss': [],'loss': [],'dice': [],'epoch':[] }  #log

#================================resume train====================================
if break_point:
    logger.info('==============using checkpoint!================')
    df = pd.read_csv(save_path + dataset_type+'.csv',sep=',')
    best_dice = df['Dice'][num_classes - 1]
    best_epoch = df['Best_eopch'][0]
    start_epoch = best_epoch
    path_checkpoint = weight_patch+str(start_epoch)+'.pth'
    checkpoint = torch.load(path_checkpoint)

    model.load_state_dict(checkpoint['model'])

    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    if train_in_test:
        df = pd.read_csv(save_path + dataset_type+'.csv',sep=',')
        max_dice = df['Dice'][num_classes-1]

#================================training====================================
logger.info('start training!')
for epoch in range(start_epoch+1,args.epochs+1):
    print('epoch',epoch)
    epoch_running_loss = 0
    train_idx = 0.0
    step_loss_ce = 0
    step_loss_dice = 0
    step_loss = 0
    trainloader_len = len(trianloader)
    model.train()
    for image, label in tqdm(trianloader):
        image = image.float()
        label = label.float()
        image, label = image.to(device), label.to(device)

        output = model(image)

        loss_ce = criterion_ce(output, label[:].long())
        loss_dice = criterion_dice(output, label, softmax=True)
        loss = loss_ce + loss_dice
        # print(loss_ce,loss_dice)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================loss====================
        step_loss_ce += loss_ce.item()
        step_loss_dice += loss_dice.item()
        #step_loss_fl += loss_fl.item()
        step_loss += loss.item()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        # ===================index====================
        # print(train_index)
    scheduler.step()
    step_loss_ce /= trainloader_len
    step_loss_dice /= trainloader_len
    step_loss /= trainloader_len
    logger.info('epoch [{}/{}], lr:{:.8f}, CrossEntropyLoss:{:.6f}, Diceloss:{:.6f}, Loss:{:.6f}'
                .format(epoch, args.epochs,lr,step_loss_ce,step_loss_dice,step_loss))

    #================================save model====================================
    checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch
        }
    if epoch % args.save_model_epoch == 0:
        logger.info('save epoch {} model'.format(epoch))
        torch.save(checkpoint, weight_patch + str(epoch) + ".pth")

    # ================================testing====================================
    # print(max_dice)
    if train_in_test:
        max_dice = val(args=args,model=model,logger=logger, epoch=epoch,\
                                    checkpoint=checkpoint,main_data_path=main_data_path,\
                                    save_path=save_path,max_dice=max_dice,weight_patch=weight_patch,\
                                    device=device,save_pred=None)

    # ================================log traning====================================
    log_dict['celoss'].append(float(step_loss_ce))
    log_dict['diceloss'].append(float(step_loss_dice))
    log_dict['loss'].append(float(step_loss))
    log_dict['epoch'].append(epoch)
    
plt.figure()
plt.plot(log_dict['epoch'],log_dict['celoss'],\
        log_dict['epoch'],log_dict['diceloss'],\
        log_dict['epoch'],log_dict['loss'])
label = ['loss_ce','loss_dice','loss']
plt.legend(label,loc='best')
record_path = main_path + 'record/'
if not os.path.exists(record_path):
    os.makedirs(record_path)
plt.savefig(record_path +data_type+'_'+loss_type+'.jpg')
