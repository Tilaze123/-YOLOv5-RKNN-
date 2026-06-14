# YOLOv5-RKNN 目标检测项目

基于 YOLOv5 的目标检测项目，支持模型训练、ONNX 导出及 RKNN（Rockchip Neural Network）部署，适用于 RK3588 等瑞芯微芯片平台的边缘推理场景。

## 项目简介

本项目基于 Ultralytics YOLOv5 框架，实现了：

- 自定义数据集的目标检测模型训练
- PyTorch 模型导出为 ONNX 格式
- ONNX 模型转换为 RKNN 格式（支持 INT8 量化）
- 在 RK3588 开发板上进行实时摄像头推理

### 支持的检测任务

| 任务 | 配置文件 | 说明 |
|------|----------|------|
| 狗脸检测 | `dogface.yaml` | 单类别：dogface |
| 人脸检测 | `fire.yaml` | 单类别：face |

## 项目结构

```
RKnn/
├── train.py            # 模型训练脚本
├── train_onnx.py       # 使用 Ultralytics 导出 ONNX 的辅助脚本
├── export.py           # YOLOv5 官方模型导出脚本
├── detect.py           # YOLOv5 官方推理脚本
├── val.py              # 模型验证脚本
├── tr.py               # RKNN 模型转换与 PC 端测试脚本
├── tr2.py              # RK3588 板端实时推理脚本
├── 锚点检测.py          # 提取训练模型锚点信息的辅助脚本
├── dogface.yaml        # 狗脸检测数据集配置
├── fire.yaml           # 人脸检测数据集配置
├── best.pt             # 训练好的 PyTorch 权重
├── best.onnx           # 导出的 ONNX 模型
├── dogface.rknn        # 转换后的 RKNN 模型
├── models/             # YOLOv5 模型定义
├── utils/              # 工具函数
├── data/               # 数据集相关文件
├── datasets/           # 训练数据集目录
├── classify/           # 分类模型相关
├── segment/            # 分割模型相关
└── requirements.txt    # Python 依赖
```

## 环境要求

### 本地训练/导出环境

- Python >= 3.7
- PyTorch >= 1.7
- 其他依赖见 `requirements.txt`

```bash
pip install -r requirements.txt
```

### RKNN 转换环境（PC 端）

```bash
pip install rknn-toolkit2
```

### RK3588 板端环境

- RK3588 开发板（如 Orange Pi 5、Rock 5A 等）
- 安装 `rknn-lite2` 运行库
- 连接摄像头（USB 或 CSI）

## 使用流程

### 1. 模型训练

使用自定义数据集进行训练：

```bash
python train.py --data dogface.yaml --weights yolov5s.pt --img 640 --epochs 100 --batch-size 16
```

参数说明：
- `--data`: 数据集配置文件路径
- `--weights`: 预训练权重（使用 `''` 从头训练）
- `--img`: 输入图像尺寸
- `--epochs`: 训练轮数
- `--batch-size`: 批次大小

### 2. 导出 ONNX 模型

**方式一：使用 Ultralytics API（推荐）**

```bash
python train_onnx.py
```

**方式二：使用官方导出脚本**

```bash
python export.py --weights best.pt --include onnx --img 640
```

### 3. 转换为 RKNN 模型

修改 `tr.py` 中的配置参数后运行：

```python
ONNX_MODEL = 'best.onnx'           # ONNX 模型路径
TARGET_PLATFORM = 'rk3588'          # 目标平台
DO_QUANT = True                     # 是否 INT8 量化
```

```bash
python tr.py
```

### 4. 提取模型锚点

部署前需要提取训练模型的实际锚点值：

```bash
python 锚点检测.py
```

将输出的锚点值填入 `tr2.py` 的 `ANCHORS` 变量中。

### 5. RK3588 板端部署

将 `dogface.rknn` 模型文件传输到开发板后运行：

```bash
python tr2.py
```

板端脚本特性：
- 使用 `rknnlite2` 库进行推理
- 支持 3 线程并行推理（充分利用 RK3588 的 3 个 NPU 核心）
- 实时摄像头检测与可视化

## 关键配置参数

### 检测参数（tr2.py）

```python
OBJ_THRESH = 0.25    # 目标置信度阈值
NMS_THRESH = 0.45    # NMS 非极大值抑制阈值
IMG_SIZE = (640, 640) # 输入图像尺寸
CAMERA_ID = 0        # 摄像头编号
THREADS = 3          # NPU 并行线程数
```

### YOLOv5 锚点

默认锚点已内置，训练自定义数据集后请使用 `锚点检测.py` 提取新锚点并更新。

## 预训练模型

项目包含以下预训练模型：

| 模型文件 | 说明 |
|----------|------|
| `yolov5s.pt` | YOLOv5s 预训练权重 |
| `yolov5s.onnx` | YOLOv5s ONNX 模型 |
| `yolov8n.pt` | YOLOv8n 预训练权重 |

## 常见问题

**Q: RKNN 转换报错怎么办？**

A: 确保 ONNX 模型的 opset 版本 >= 12，使用 `simplify=True` 简化模型。

**Q: 板端推理速度慢？**

A: 确认开启了 INT8 量化（`DO_QUANT=True`），并使用 3 线程并行（`THREADS=3`）。

**Q: 如何修改检测类别？**

A: 修改对应的 yaml 配置文件中的 `names` 字段，重新训练模型。

## 参考资料

- [YOLOv5 官方仓库](https://github.com/ultralytics/yolov5)
- [RKNN Toolkit2 文档](https://github.com/airockchip/rknn-toolkit2)
- [RKNN 模型转换指南](./README_rkopt.md)

## 许可证

本项目基于 [GPL-3.0 许可证](LICENSE) 开源。
