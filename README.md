# SPU-IMR: Self-supervised Arbitrary-scale Point Cloud Upsampling via Iterative Mask-recovery Network
## Environment
```
Pytorch >= 1.9.0
cuda >= 11.1
```

## Training
```
python train.py
```
## Dataset
The dataset we used for training and testing is obtained by [SAPCU](https://github.com/xnowbzhao/sapcu).
```
https://pan.baidu.com/s/1OPVnCHq129DBMWh5BA2Whg 
access code: hgii

or

https://1drv.ms/f/s!AsP2NtMX-kUTml4U3DYUD6Hy9FJn?e=8QfJTH
```

## Evaluation
### a. Download models
Download the pretrained models from the link and unzip it to ./out/
```
Link: https://pan.baidu.com/s/1ku7y7fLgJlsreTIx1Zlb4w?pwd=zxys
Access code: zxys 
```
### b. Evaluation
```
cd eval
python evaluation.py
```
### c. Points to note
More code for evaluation can be downloaded from:
```
https://github.com/pleaseconnectwifi/Meta-PU/tree/master/evaluation_code
https://github.com/jialancong/3D_Processing
```
## Acknowledgement
The code is based on [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), [SAPCU](https://github.com/xnowbzhao/sapcu) and [Point-Transformer](https://openaccess.thecvf.com/content/ICCV2021/html/Zhao_Point_Transformer_ICCV_2021_paper.html?ref=;), Thanks for their great work. If you use any of this code, please cite these works.
