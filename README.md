# 👾 SEG-YOLO
![](https://img.shields.io/badge/keras-tensorflow-blue.svg)
![](https://img.shields.io/badge/python-c-green.svg)
> This is the instruction of how to use the foreground detection code, including data pre-processing, training, evaluation and inference.

### Prerequisites
* Ubuntu 16.04
* **CMake >= 3.8** for modern CUDA support: https://cmake.org/download/
* **CUDA 10.0**: https://developer.nvidia.com/cuda-toolkit-archive (on Linux do [Post-installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions))
* **cuDNN >= 7.0 for CUDA 10.0** https://developer.nvidia.com/rdp/cudnn-archive (on **Linux** copy `cudnn.h`,`libcudnn.so`... as desribed here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar)
* **GPU with CC >= 3.0**: https://en.wikipedia.org/wiki/CUDA#GPUs_supported
* on Linux **GCC or Clang**
* Python3.6

### Getting Started
everytime change the syetem, open the terminal:
```shell
cd darknet
make clean
make
```
copy 'libdarknet.so' to directory 'SEG-YOLO/src'
```shell
workon py36
```
All files are based on the root foreground_detection
YOLOv3 version and tutorial: https://github.com/AlexeyAB/darknet

### Step0: Self annotation to output coco format segmenatation data:
https://github.com/jsbroks/coco-annotator
Annotation interface and result on localhost:5000

### Step1: Train darknet yolov3 on COCO subset(train on categories backpack and person to gain a general feature learner)
1. Prepare COCO data:
(1) put 'COCO2014' directory in the 'foreground_detection' directory, this is the COCO2014 contains subset's label which different from the original one.
(2) command to generate train.txt and valid.txt
```shell
cd COCO
# Set Up Image Lists
paste <(awk "{print \"$PWD\"}" <5k.part) 5k.part | tr -d '\t' > 5k.txt
paste <(awk "{print \"$PWD\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt
```
(3) copy '5k.txt' and 'trainvolno5k.txt' to directory 'darknet/coco_subset/cfg'

2. train on COCO subset:
```shell
cd darknet
export CUDA_VISIBLE_DEVICES=0,1,2
./darknet detector train coco_subset/cfg/coco_subset.data coco_subset/cfg/yolov3_coco_subset.cfg darknet53.conv.74 -gpus 0,1,2
```
weights in 'coco_subset/weights'

### Step2: Fine-tune the yolov3 weights using golfer/golfbag dataset
1. Find the yolov3 weights trained on COCO subset with the highest mAP:
```shell
./darknet detector map coco_subset/cfg/coco_subset.data coco_subset/cfg/yolov3_coco_subset.cfg coco_subset/weights/xxxxxxx.weights
```

2. Prepare self-annotated data:
In segmentation_data, .zip/.tar.gz file is the images and .json file is the annotation, images number>annotation number
First/First1.json for Video_sample1, Second.json for Video_sample2, etc. The Fourth.json and Video_sample4 is not for yolo training.
Thus totally 2500 images for yolov3 fine-tuning. Background.zip only contains some amout of background data which is labeled with nothing.(An empty '.txt' file)
(1) command
```shell
git clone https://github.com/ssaru/convert2Yolo
cd Convert2Yolo
# create .txt file for each annotated images in Video_Sample1/2/3
python example.py --datasets COCO --img_path ~/foregroud_detection/segmentation_data/Video_Sample1/ --label ~/foregroud_detection/segmentation_data/First.json --convert_output_path ~/foregroud_detection/segmentation_data/label_First/ --img_type ".jpg" --manipast_path ~/foregroud_detection/segmentation_data/label_First/ --cls_list_file ~/foregroud_detection/darknet/fine_tune_2500/cfg/fine_tune.names 
python example.py --datasets COCO --img_path ~/foregroud_detection/segmentation_data/Video_Sample1/ --label ~/foregroud_detection/segmentation_data/First1.json --convert_output_path ~/foregroud_detection/segmentation_data/label_First1/ --img_type ".jpg" --manipast_path ~/foregroud_detection/segmentation_data/label_First1/ --cls_list_file ~/foregroud_detection/darknet/fine_tune_2500/cfg/fine_tune.names 
python example.py --datasets COCO --img_path ~/foregroud_detection/segmentation_data/Video_Sample2/ --label ~/foregroud_detection/segmentation_data/Second.json --convert_output_path ~/foregroud_detection/segmentation_data/label_Second/ --img_type ".jpg" --manipast_path ~/foregroud_detection/segmentation_data/label_Second/ --cls_list_file ~/foregroud_detection/darknet/fine_tune_2500/cfg/fine_tune.names 
python example.py --datasets COCO --img_path ~/foregroud_detection/segmentation_data/Video_Sample3/ --label ~/foregroud_detection/segmentation_data/Third.json --convert_output_path ~/foregroud_detection/segmentation_data/label_Third/ --img_type ".jpg" --manipast_path ~/foregroud_detection/segmentation_data/label_Third/ --cls_list_file ~/foregroud_detection/darknet/fine_tune_2500/cfg/fine_tune.names 
```
(2) intergret all the manifast.txt to train.txt and valid.txt
(3) create empty .txt file for the background data and add the path into the train/valid.txt
(4) copy train.txt and valid.txt to the 'darknet/fine_tune_2500/cfg/'
(5) put .txt files and .jpg files that are related together, which is copy the .txt file to the Video_Sample folder.

3. Re-calculate the anchor of the yolov3 by:
```shell
cd darknet
./darknet detector calc_anchors fine_tune_2500/cfg/fine_tune.data -num_of_clusters 9 -width 320 -height 320
```
copy the anchors result to the yolov3_fine_tune_2500.cfg each 'anchors' value in each [yolo] block.

4. Train the final yolov3 model for golfer and golfbag
```shell
./darknet detector train fine_tune_2500/cfg/fine_tune.data fine_tune_2500/cfg/yolov3_fine_tune_2500.cfg fine_tune_2500/yolov3_COCO_best.weights -gpus 0,1,2
```

5. Find the final yolov3 weights with highest mAP (Iteriation 87000):
```shell
./darknet detector map fine_tune_2500/cfg/fine_tune.data fine_tune_2500/cfg/yolov3_fine_tune_2500.cfg fine_tune_2500/weights/xxxxx.weights
```

### Step3: Train the FCN head based on the fine-tune YOLOv3 weights:
```shell
cd SEG-YOLO
``` 
1. Copy the final yolov3.cfg and yolov3.weights to 'SEG-YOLO' directory
2. Run convert.py to transform the yolov3.weights into keras '.h5' format:
python convert.py cfg/yolov3.cfg  yolov3.weights yolov3.h5
3. Change src/masknet.py my_num_rois=64 for training
4. Run train_coco.py to first train FCN on COCO subset.
5. Run train.py to fine-tune the FCN head on self-annotated dataset.
6. Change src/masknet.py my_num_rois=8 for detection.
5. Run detection.py to see the result:
```shell
pyhon detection.py test_video/xxxxx.mov
```

### Step4: Change model size for speed-accuracy trade off
1. Change yolo detection size to 416x416 for better object detection, need to re-train the FCN by setting src/masknet.py my_net_inp = 416
2. Change mask size to 28 for better mask prediction, now is mask size 14 with higher speed but rough boundary. Change weights in detection.py:
model.load_weights("./weights/mask14_direct_75.hdf5")
in 'SEG-YOLO/weights/' 14 denotes mask size, direct for training on self-annotated dataset, 75 for iteration(iteration 100 have overfitting problem I recommand 75).
Also change 'src/masknet.py' my_msk_inp = 14
