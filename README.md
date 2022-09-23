# EagerMOT

这是一个使用MindSpore框架的 EagerMOT 算法。

## 说明

这是一个使用 MindSpore 框架 EagerMOT 算法的重构版本，增加了搜索超参数的函数 train.py
原版的论文为 EagerMOT: 3D Multi-Object Tracking via Sensor Fusion。 [链接](https://github.com/aleksandrkim61/EagerMOT)

## 所需环境

- `python==3.7`
- `filterpy==1.4.5`
- `glob2==0.6`
- `more-itertools==7.2.0`
- `numba==0.46.0`
- `numpy==1.18`
- `nuscenes-devkit==1.1.0`
- `opencv-python==4.4.0.42`
- `Pillow==6.2.0`
- `pyquaternion==0.9.5`
- `scipy==1.2.0`
- `Shapely==1.7.0`

## 文件结构

```

${root_dir}
|-- dataset
`-- |-- base
    `-- |-- mot_dataset.py  # dataset 基类，负责提供 sequence 地址索引
        |-- mot_sequence.py # sequence 基类，负责结果的读取保存，算法输入为 sequence
	|-- mot_frame.py    # frame 基类，负责每一帧检测融合
`-- |-- kitti_dataset.py    # 上述三个类的 kitti 版本子类
`-- |-- nuscenes_dataset.py # 上述三个类的 nuscenes 版本子类
|-- models
`-- |-- backbones
    `-- |-- Eagermot.py	    # 算法进行的主体 API
`-- |-- blocks
    `-- |-- TrackManager.py # 跟踪管理模块
    `-- |-- match_3d.py     # 第一阶段 3D 匹配
    `-- |-- match_2d.py     # 第二阶段 2D 匹配
`-- |-- run_tracking.py     # 算法运行开始
`-- |-- train.py 	    # 超参数遍历
|-- utils                   # 一些基础算法函数


```

## 数据集准备

下载 Kitti 或 NuScenes 数据集，并下载所需的 2D 和 3D 检测结果
KITTI 2D MOTSFusion detections/segmentations [链接](https://github.com/tobiasfshr/MOTSFusion) 在 "Results" 下面，有相应的检测结果
KITTI 2D TrackRCNN detections/segmentations [链接](https://www.vision.rwth-aachen.de/page/mots) 在 "Downloads" 下面，有相应的检测结果
KITTI 3D PointGNN, NuScenes 3D CenterPoint, NuScenes 2D detections 的检测结果在这个[链接](https://drive.google.com/drive/folders/1MpAa9YErhAZNEJjIrC4Ky21YfNj2jatM)下面

## 测试结果

### Kitti

Car

| | HOTA   | DetA   | AssA   | MOTA   |
| ----- | ------ | ------ | ------ | ------ |
|MindSpore版| 78.037 | 76.802 | 79.515 | 87.246 |
| 原版 |74.39|75.27|74.16|87.82|

Pedestrian

| | HOTA   | DetA   | AssA   | MOTA   |
| ----- | ------ | ------ | ------ | ------ |
|MindSpore版| 48.196 | 48.989 | 47.812 | 62.166 |
|原版|39.38|40.60|38.72|49.82|

### Nuscenes

| | AMOTA | MOTA | IDs  |
| ----- | ----- | ----- | ---- |
|MindSpore版| 0.676 | 0.568| 1156 |
|原版|0.68|0.57|1156|

Nuscenes 测试结果在：链接：https://pan.baidu.com/s/1uBITCC3G1jChynkkoxhiew 提取码：rd1e

kitti 测试结果在：链接：https://pan.baidu.com/s/1KjygxSqpjGlBny0waoMueQ 提取码：2lvf，或 utils/test_result/kitti

## 使用说明

预测：
```
python run_tracking.py
```
可以根据 Args 更换数据集和对应的SPLIT


参数搜索：
```
python train.py
```
## 参考文献

```
@inproceedings{Kim21ICRA,
  title     = {EagerMOT: 3D Multi-Object Tracking via Sensor Fusion},
  author    = {Kim, Aleksandr, O\v{s}ep, Aljo\v{s}a and Leal-Taix{'e}, Laura},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2021}
}
```
