# 图像中血管分割任务

## 数据集
1. The Digital Retinal Images for Vessel Extraction (DRIVE) database(原图像，标记图像，轮廓mask图像分别40张)

## 数据处理
- 训练
    - 原图像：(20,3,584,565)， 标记图像：(20,1,584,565)
    - x_train：(190000,1,48,48)， label：(190000,2304,2)
- 验证
    - 训练数据的10%
- 测试
    - 原图像：(20,3,584,565)， 标记图像：(20,1,584,565)
    - x_test：(58300,1,48,48)

## 评价指标
   - 混淆矩阵
   - AUC
   - PRC
   - F1-score
   - Jaccard similarity score
   - ACCURACY
   - SENSITIVITY
   - SPECIFICITY
   - PRECISION
   - 单张图片处理速度
  
|  指标 \| 网络 | unet | vssc| vssc_class_weigth|
|  ----  | ----  | ---   | ---   |
|  AUC   | 0.977 | 0.977 | 0.977 |
|  PRC   | 0.905 | 0.906 | 0.903 |
|  F1    | 0.806 | 0.809 | 0.763 |
|  JSS   | 0.954 | 0.955 | 0.927 |
|  ACC   | 0.954 | 0.955 | 0.927 | 
|  SN    | 0.744 | 0.750 | 0.927 |
|  SP    | 0.985 | 0.985 | 0.926 |
|  PR    | 0.880 | 0.878 | 0.648 |

## 测试结果
1. unet
![1](/home/liux/文档/md/unet_true_pred.png) 

2. vssc
![1](/home/liux/文档/md/vssc_true_pred.png)

3. vssc_clas_weight
![1](/home/liux/文档/md/vssc_class_weight_true_pred.png)
