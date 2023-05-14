# ASF-LKUNet

The code of ASF-LKUNet from [ASF-LKUNet: Adjacent-Scale Fusion U-Net with Large-kernel for Medical Image Segmentation](https://www.techrxiv.org/articles/preprint/ASF-LKUNet_Adjacent-Scale_Fusion_U-Net_with_Large-kernel_for_Medical_Image_Segmentation/22794728)

The code of ASF-LKUNet can be used for academic research only, please do not use them for commercial purposes. If you have any problem with the code, please contact: rfwang@xidian.edu.cn or mzhaoshan@163.com.

lf you think this work is helpful, please cite：
Wang, Rongfang; Mu, zhaoshan; Wang, kai; Liu, Hui; Zhou, Zhiguo; Gou, Shuiping; Wang, Jing; and Jiao, Licheng;(2023): ASF-LKUNet: Adjacent-Scale Fusion U-Net with Large-kernel for Medical Image Segmentation. TechRxiv. Preprint. https://doi.org/10.36227/techrxiv.22794728.v1

### 1. Prepare data

- Synapse dataset can be found at [here](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789).

- ACDC dataset can be found at [here](https://www.creatis.insa-lyon.fr/Challenge/acdc/). 

- Data Preparing

  1. Synapse and ACDC images to the resolution 1 × 1 × 3  and 1.56 × 1.56 × 3. We normalized the CT values of in the Synapse dataset by min-max normalization to [0, 1]. The min-max values were set to -125 and 275. The ACDC dataset by z-score normalization.

  2. 2.5D input is $[i-1, i, i+1]$, When $i=1$ or $i=i_{\max }$, the stack is $[1, 1, i+1]$ or $[i-1, i_{\max }, i_{\max }]$

  3. The dataset directory structure of the whole project is as follows:
  
     ```
     ├── ASF_LKUNet
     │   └──...
     └── data
         └──Synapse(or ACDC)
             └──test
             	└──image
                     │   ├── 0.npy
                     │   ├── 1.npy
                     │   ├── ...
                     │   └── *.npy
                 └──masks
                     │   ├── 0.npy
                     │   ├── 1.npy
                     │   ├── ...
                     │   └── *.npy
             └── train
            		└──image
                     │   ├── 0.npy
                     │   ├── 1.npy
                     │   ├── ...
                     │   └── *.npy
                 └──masks
                     │   ├── 0.npy
                     │   ├── 1.npy
                     │   ├── ...
                     │   └── *.npy
     ```
     
     

### 2. Environment

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 3.Train/Test

#### Train

- Synapse dataset or ACDC

```bash
python train_sy.py/train_acdc.py --data_path 'your data path' --root_path 'your main path' 
```

#### Test

- Synapse dataset or ACDC

```
python test_sy.py/test_acdc.py --data_path 'your data path' --root_path 'save model path' 
```



## Reference
* [TransUnet](https://github.com/Beckschen/TransUNet?utm_source=catalyzex.com)

* [ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)

  

