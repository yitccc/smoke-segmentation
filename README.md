# EGNL-FAT: An Edge-Guided Non-Local Network with Frequency-Aware Transformer for Smoke Segmentation

Yitong Fu, Haiyan Li, Yujiao Wang, Wenbing Lv, Bingbing He, Pengfei Yu


![Image](https://github.com/yitccc/smoke-segmentation/blob/master/images/1.png)

   EGNL-FAT effectively segments smoke in real-world scenarios while maintaining a balance between accuracy and computational efficiency. Based on 256×256 inputs, it achieves an inference speed of 15.27 fps with only 11.11 GFLOPs and a model size of 19.72M parameters. Experimental results on the FSS dataset show that the proposed method achieves mean IoUs of 59.66% on FS01 and 64.68% on FS02, while further reaching 79.04% on SMOKE5K, demonstrating its robustness in diverse smoke environments.

## Datasets
   You can download cityscapes from [here](https://github.com/yitccc/smoke-segmentation/tree/master/FSS).It should have this basic structure.

```
├── data
│   ├── build256
│   │   ├── train
│   │   │   ├── images
│   │   │   └── labels
│   │   ├── val
│   │   │   ├── images
│   │   │   └── labels
│   │   ├── test
│   │   │   ├── images
│   │   │   └── labels

```
   
## Results
FSS
![Image](https://github.com/yitccc/smoke-segmentation/blob/master/images/5.png)
FS01
![Image](https://github.com/yitccc/smoke-segmentation/blob/master/images/2.png)

FS02
![Image](https://github.com/yitccc/smoke-segmentation/blob/master/images/3.png)

SMOKE5K
![Image](https://github.com/yitccc/smoke-segmentation/blob/master/images/6.png)
![Image](https://github.com/yitccc/smoke-segmentation/blob/master/images/4.png)


