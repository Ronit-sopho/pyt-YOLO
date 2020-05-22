## YOLOv3 PyTorch

#### PyTorch implementation of YOLOv3

**How To's**

**Inference**
- Download pretrained yolov3 weights in pytorch format [here](https://drive.google.com/open?id=1EfxwYU15WcQsqIdYxkXnMygk3TUojb7F) **OR** Convert darknet weights using:
```
python extras.py --convert path_to_cfg_file path_to_weights_file
```
and place them in logs directory
- Run inference using infer.py
```
python infer.py --image path_to_image --conf_thresh 0.7 --nms_thresh 0.5 --model path_to_pt_file
```
**Task List**
 - [ ] Fix training pipeline
 - [ ] Add comments

**Environment used**
- pytorch 1.1.0
- numpy 1.16.2
- python 3.6.9