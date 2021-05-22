# TT-SPN: Twin Transformers with Sinusoidal Representation Networks for Video Instance Segmentation

Object Detection and Instance Segmentation version of TT-SPN

Video instance segmentation is the recently introduced computer vision research are that aims joint detection, segmentation and tracking of instances in the video domain. Recent methods proposed highly sophisticated and multi-stage networks that lead to be unusable in practise. Hence, simple yet effective single stage approaches are needed to be used in practise. To fill the gap, we propose end-to-end transformer based video instance segmentation module with Sinusoidal Representation Networks (SPN), namely TT-SPN, to address this problem. TT-SPN, views the VIS task as direct sequence prediction problem in single stage that enables us to aggregate temporal information with spatial one.

![TT-SPN](https://github.com/cankocagil/TT-SPN/blob/main/figures/Pipeline.png?raw=true)


 Set of video frame features are extracted by twin transformers that then propagated to original transformer to produce sequence of instance predictions. These produced instance level information by transformers are then passed through modified Sinusoidal Representation Networks to get end instance level class ids and bounding boxes and self-attended convolutions to get segmentation masks. At its core, TT-SPN is natural paradigm that handles the instance segmentation and tracking via similarity learning that enables system to produce fast and accurate set of predictions. TT-SPN is trained end-to-end with set-based global loss that forces unique predictions via bipartite matching. Thus, general complexity of pipeline is significantly decreased without sacrificing quality of segmentation masks. For the first time, VIS problem is addressed without implicit CNN architectures thanks to twin transformers with being one of the fastest approaches. Our method can be easily divided into its sub-components to produce separate instance masks and bounding boxes that will make it unified approach for many vision tasks.  We benchmark our results on YouTube-VIS dataset by comparing competitive baselines and show that TT-SPN outperforms the base VIS model by significant margin.


# Installation

We provide installation quidelines for TT-SPN. 
First, clone our project page as follows.

```
git clone https://github.com/cankocagil/TT-SPN---Object-Detection
```

Then, install PyTorch 1.5+ and torchvision 0.6+:
```
conda install -c pytorch pytorch torchvision
```
Install pycocotools (for evaluation on COCO) and scipy (for training):
```
conda install cython scipy
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
# Data Preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

## Training
To train baseline DETR on a single node with n gpus for 300 epochs run:
```
python -m torch.distributed.launch --nproc_per_node=n --use_env main.py --coco_path /path/to/coco 
```


# Acknowledgement
We would like to thank the [VisTR](https://github.com/Epiphqny/VisTR) and [DETR](https://github.com/facebookresearch/detr) open-source projects for their awesome work, part of the code are modified from their projects.



