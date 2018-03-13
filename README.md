# Visual Relationship Detection with Deep Structural Ranking

The code is written in python and pytorch (0.2.0).

### Data Preparation

1. Download [VRD Dateset](http://cs.stanford.edu/people/ranjaykrishna/vrd/dataset.zip) and put it in the path ~/data. Replace ~/data/sg_dataset/sg_test_images/4392556686_44d71ff5a0_o.gif with ~/data/vrd/4392556686_44d71ff5a0_o.jpg

2. Download [VGG16 trained on ImageNet](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM) and put it in the path ~/data

3. Download [the meta data (so_prior.pkl)](https://pan.baidu.com/s/1qZErdmc) and put it in ~/data/vrd

4. Download [visual genome data (vg.zip)](https://pan.baidu.com/s/1qZErdmc) and put it in ~/data/vg

The folder should be:

    ├── sg_dataset
    │   ├── sg_test_images
    │   ├── sg_train_images
    │   
    ├── VGG_imagenet.npy
    └── vrd
        ├── gt.mat
        ├── obj.txt
        ├── params_emb.pkl
        ├── proposal.pkl
        ├── rel.txt
        ├── so_prior.pkl
        ├── test.pkl
        ├── train.pkl
        └── zeroShot.mat

### Prerequisites

* Python 2.7
* Pytorch 0.2.0
* opencv-python
* tabulate
* CUDA 8.0 or higher

### Installation 

* Edit ~/lib/make.sh to set CUDA_PATH and choose your `-arch` option to match your GPU.

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |
  
* Build the Cython modules for the roi_pooling layer and choose the right -arch to compile the cuda code refering to https://github.com/ruotianluo/pytorch-faster-rcnn.

    ```bash
    cd lib
    ./make.sh
    ```

### Train

* Model Structure

![Model Structure](https://github.com/GriffinLiang/vrd-dsr/blob/master/img/net.png)

* CUDA_VISIBLE_DEVICES=0 python train.py --dataset vrd --name VRD_RANK --epochs 10 --print-freq 500 --model_type RANK_IM

* This project contains all training and testing code for predicate detection. For relationship detection, our proposed pipeline contains two stages. The first stage is object detection and not included in this project. I am trying to release the code ASAP. Before that, you may refer to some other projects such as [pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn) and [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch).

## Citation

If you use this code, please cite the following paper(s):

	@article{liang2018Visual,
		title={Visual Relationship Detection with Deep Structural Ranking},
		author={Liang, Kongming and Guo, Yuhong and Chang, Hong and Chen, Xilin},
  		booktitle={AAAI Conference on Artificial Intelligence},
  		year={2018}
	}

## License

The source codes and processed data can only be used for none-commercial purpose. 
