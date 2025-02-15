# POL: Pointed Object Localization
POL is used to determine the pose of the object that is being pointed by the human. This project is build as a ROS package and uses YOLO-v7 models for human pose detection and instance segmentation. The algorithm is based on this [paper](https://drive.google.com/file/d/1Jyk4RMjvvgMNGzvr7nbpZMZ5aRgOA6Ac/view?usp=share_link)

## Requirements 
- ZED camera is connected
- ZED ROS Wrapper is running
- Model weights are downloaded. Run the following command to download the weights
```
cd weights
bash download_weights.sh
```

## Testing
Source the package
```
source ~/catkin_ws/devel/setup.bash
```
### Run the human pose detection node
```
cd scripts/yolov7_pose
python3 pose_estimate.py
```

### Run the instance segmentation node
```
cd scripts/yolov7_segmentation
python3 segmentation.py
```
