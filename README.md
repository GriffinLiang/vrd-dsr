# Visual Relationship Detection with Deep Structural Ranking

The code is written in python and pytorch (0.2.0) [torch-0.2.0.post3].

Since I have graduated, I may not be able to respond to the issues in time. Thanks for your understanding.

### Clone the repo
  * `git clone git@github.com:GriffinLiang/vrd-dsr.git`
  * `git submodule update --recursive`

  OR
  * `git clone --recursive git@github.com:GriffinLiang/vrd-dsr.git`

### Data Preparation

1. Download VRD Dateset ([image](http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip), [annotation](http://cs.stanford.edu/people/ranjaykrishna/vrd/dataset.zip), [backup](https://drive.google.com/drive/folders/1V8q2i2gHUpSAXTY4Mf6k06WHDVn6MXQ7)) and put it in the path ~/data. Replace ~/data/sg_dataset/sg_test_images/4392556686_44d71ff5a0_o.gif with ~/data/vrd/4392556686_44d71ff5a0_o.jpg

2. Download [VGG16 trained on ImageNet](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM) and put it in the path ~/data

3. Download the meta data (so_prior.pkl) [[Baidu YUN]](https://pan.baidu.com/s/1qZErdmc) or [[Google Drive]](https://drive.google.com/open?id=1e1agFQ32QYZim-Vj07NyZieJnQaQ7YKa) and put it in ~/data/vrd

4. Download visual genome data (vg.zip) [[Baidu YUN]](https://pan.baidu.com/s/1qZErdmc) or [[Google Drive]](https://drive.google.com/open?id=1QrxXRE4WBPDVN81bYsecCxrlzDkR2zXZ) and put it in ~/data/vg

5. Word2vec representations of the subject and object categories are provided in this project. If you want to use the model for novel categories, please refer to this [blog](http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/).

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
### Data format

* train.pkl or test.pkl
	* python list
	* each item is a dictionary with the following keys: {'img_path', 'classes', 'boxes', 'ix1', 'ix2', 'rel_classes'}
	  * 'classes' and 'boxes' describe the objects contained in a single image.
	  * 'ix1': subject index.
	  * 'ix2': object index.
	  * 'rel_classes': relationship for a subject-object pair.


* proposal.pkl
	```Python
        >>> proposals.keys()
        ['confs', 'boxes', 'cls']
        >>> proposals['confs'].shape, proposals['boxes'].shape, proposals['cls'].shape
        ((1000,), (1000,), (1000,))
        >>> proposals['confs'][0].shape, proposals['boxes'][0].shape, proposals['cls'][0].shape
        ((9, 1), (9, 4), (9, 1))
        ```

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
    
### Demo
 * Predicate demo: demo.py->pre_demo()
   * Download epoch_4_checkpoint.pth.tar [[Baidu YUN]](https://pan.baidu.com/s/1POE2LKJulOoHqEkWV-XHig) or [[Google Drive]](https://drive.google.com/file/d/1_jWnvWNwlJ2ZqKbDMHsSs4BjTblg0FSe/view?usp=sharing) and put it in ~/model
 * Relationship demo: demo.py->vrd_demo().
   * Install [faster-rcnn](https://github.com/GriffinLiang/faster-rcnn.pytorch/tree/773184a60635918e43b320eb1a0e8881779b90c8
) according to  README file. (Pay attention to ~/lib/make.sh. Set CUDA_PATH by choosing your `-arch` option to match your GPU.)

   * Download faster_rcnn_1_20_7559.pth [[Baidu YUN]](https://pan.baidu.com/s/1V0QIiEI06tcKQOTcHkaorQ) or [[Google Drive]](https://drive.google.com/file/d/11YQ7Ctj7kaau6WTx5MKkbw6PIxJAyvsZ/view?usp=sharing) and put it in ~/model
   * [Thanks Jianwei Yang and Jiasen Lu for the detector codes!](https://github.com/jwyang/faster-rcnn.pytorch)
   
### Train

* Model Structure

![Model Structure](https://github.com/GriffinLiang/vrd-dsr/blob/master/img/net.png)

* Run

  ```bash
  cd tool
  CUDA_VISIBLE_DEVICES=0 python train.py --dataset vrd --name VRD_RANK --epochs 10 --print-freq 500 --model_type RANK_IM
  ```
  
  You can set the parser argument -no_so to discard separate bbox visual input and --no_obj to discard semantic cue.

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
