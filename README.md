# ASF_LKUNet

This repo holds code for ASF-LKUNet: Adjacent-Scale Fusion U-Net with Large-kernel for Medical Image Segmentation

### 1. Prepare data

- Synapse dataset can be found at [here](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789).

- ACDC dataset can be found at [here](https://www.creatis.insa-lyon.fr/Challenge/acdc/). 

- Data Preparing

  1. Synapse and ACDC images to the resolution 1 × 1 × 3  and 1.56 × 1.56 × 3. We normalized the CT values of in the Synapse dataset by min-max normalization to [0, 1]. The min-max values were set to -125 and 275. The ACDC dataset by z-score normalization.

  2. 2.5D input:$[i-1, i, i+1]$, When $i=1$ or $i=i_{\max }$, the stack is $[1, 1, i+1]$ or $[i-1, i_{\max }, i_{\max }]$

  3. The dataset directory structure of the whole project is as follows:
  
     ```
     ├── ASF_LKUNet
     │   └──...
     └── data
         └──Synapse
             ├── test
             │   ├── 0.npy
             │   ├── 1.npy
             │   ├── ...
             │   └── *.npy
             └── train
             │   ├── 0.npy
             │   ├── 1.npy
             │   ├── ...
             │   └── *.npy
         └──ACDC
             ├── test
             │   ├── 0.npy
             │   ├── 1.npy
             │   ├── ...
             │   └── *.npy
             └── train
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
