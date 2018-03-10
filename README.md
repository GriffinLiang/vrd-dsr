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

### Train 

* Build the Cython modules for the roi_pooling layer and choose the right -arch to compile the cuda code refering to https://github.com/ruotianluo/pytorch-faster-rcnn.

    ```bash
    cd lib
    ./make.sh
    ```

* CUDA_VISIBLE_DEVICES=0 python train.py --dataset vrd --name VRD_RANK --epochs 10 --print-freq 500 --model_type RANK_IM

## Citation

If you use this code, please cite the following paper(s):

	@article{liang2018Visual,
		title={Visual Relationship Detection with Deep Structural Ranking},
		author={Liang, Kongming and Guo, Yuhong and Chang, Hong and Chen, Xilin},
  		booktitle={AAAI Conference on Artificial Intelligence},
  		year={2018}
	}
