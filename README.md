# ASF_LKUNet
### Train

- Synapse dataset or ACDC

```bash
python train_sy.py --data_path 'your data path' --root_path 'your main path' 
```

```
python train_acdc.py --data_path 'your data path' --root_path 'your main path' 
```

### Test

- Synapse dataset or ACDC

```
python test_sy.py --data_path 'your data path' --root_path 'save model path' 
```

```
python test_acdc.py --data_path 'your data path' --root_path 'save model path' 
```



## Reference
* [TransUnet](https://github.com/Beckschen/TransUNet?utm_source=catalyzex.com)
* [ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)
