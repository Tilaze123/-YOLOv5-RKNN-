import torch
from models.yolo import Model

weights = 'runs/train/exp29/weights/best.pt'  # 你的训练权重路径
ckpt = torch.load(weights, map_location='cpu')
# 方法1：直接从模型属性获取
model = ckpt['model'].float()
anchors = model.model[-1].anchors.cpu().numpy()  # shape: (3, 3, 2)
print(anchors.tolist())   # 直接复制这个输出到 tr2.py 的 ANCHORS 里
