# PulsarStar
Machine learning and DeepLearning demo project about Pulsar Star

## SVM Classifier

> https://www.kaggle.com/code/prashant111/svm-classifier-tutorial/notebook
> https://as595.github.io/HTRU1/

## Dependencies

> https://github.com/kaggle/docker-python

docker image: https://github.com/kaggle/docker-python

## DateSet

- https://www.kaggle.com/datasets/colearninglounge/predicting-pulsar-starintermediate
- csv：https://archive.ics.uci.edu/ml/datasets/HTRU2
- images：https://as595.github.io/HTRU1/

## Code Details

- pytorch 1.14.0.dev20221030 for MAC M1 GPU (mps)
- dataset will be downloaded in `/data` dictionary
- trained neural network models will be saved to `model` dictionary
- svm methods is placed in `src/svm` dictionary
  - linear kernel
  - polynomial kernel
  - rbg kernel
  - sigmoid kernel
  - gridsearch method for best parameters
  - train.py: common method used to train svm model
- deeplearning methods is places in `src/deeplearning` dictionary
  - ann.py: common methods used to train and predict in deeplearning
  - pulsar.py: csv data set loading function
  - cifar10.py: image data set loading  function
  - cnn.py: common cnn demo
  - cnnimg: image cnn demo function copied from https://as595.github.io/HTRU1/
  - res14.py: resnet neural network for csv data
  - resimg14.py: resnet neural network for image data

## Known Issues

- mps device not works on csv data : share files not work fine for csv data loader
- cnnimg.py not works fine while showing example image

## References

- [SVM Classifier Tutorial](https://www.kaggle.com/code/prashant111/svm-classifier-tutorial)
- [天眼科学目标：脉冲星的观测与研究意义](https://www.cas.cn/kx/kpwz/201710/t20171016_4617668.shtml)
- [脉冲星候选样本分类方法综述：中国科学院 天文大数据中心](http://jdse.bit.edu.cn/fileSKTCXB/journal/article/sktcxb/2018/3/PDF/20180301.pdf)
- [脉冲星数据比对分析和可视化系统设计与实现](https://fast.bao.ac.cn/static/uploadfiles/FastPaper/11-脉冲星数据比对分析和可视化系统设计与实现.pdf)
- [Undersampling and oversampling imbalanced data](https://www.kaggle.com/code/residentmario/undersampling-and-oversampling-imbalanced-data)
- [Undersampling and oversampling imbalanced data](https://as595.github.io/HTRU1/)
- [HTRU2 Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2)
- [ResNet](https://arxiv.org/abs/1512.03385)