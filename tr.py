import os
import cv2
import numpy as np
from rknn.api import RKNN

# ==========================================
# 1. 基础参数配置 (请根据实际情况修改)
# ==========================================
ONNX_MODEL = '../model/yolov5s.onnx' 
DATASET_TXT = 'dataset.txt'  # 里面只有一行测试图片的路径
TEST_IMG = '../model/dog_224x224.jpg' # 测试图片路径
TARGET_PLATFORM = 'rk3588'    # 目标芯片型号
DO_QUANT = True               # 是否进行 INT8 量化 (改为 False 即为 FP 浮点测试)

# ==========================================
# 2. YOLOv5 后处理常量配置
# ==========================================
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)  # (width, height)

# 默认的 YOLOv5 anchors
ANCHORS = [[[10.0, 13.0], [16.0, 30.0], [33.0, 23.0]], 
           [[30.0, 61.0], [62.0, 45.0], [59.0, 119.0]], 
           [[116.0, 90.0], [156.0, 198.0], [373.0, 326.0]]]

CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop","mouse","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")

# ==========================================
# 3. YOLOv5 后处理函数 (来自官方 yolov5.py)
# ==========================================
def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    return boxes, classes, scores

def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def box_process(position, anchors):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    col = col.repeat(len(anchors), axis=0)
    row = row.repeat(len(anchors), axis=0)
    anchors = np.array(anchors)
    anchors = anchors.reshape(*anchors.shape, 1, 1)

    box_xy = position[:,:2,:,:]*2 - 0.5
    box_wh = pow(position[:,2:4,:,:]*2, 2) * anchors

    box_xy += grid
    box_xy *= stride
    box = np.concatenate((box_xy, box_wh), axis=1)

    xyxy = np.copy(box)
    xyxy[:, 0, :, :] = box[:, 0, :, :] - box[:, 2, :, :]/ 2
    xyxy[:, 1, :, :] = box[:, 1, :, :] - box[:, 3, :, :]/ 2
    xyxy[:, 2, :, :] = box[:, 0, :, :] + box[:, 2, :, :]/ 2
    xyxy[:, 3, :, :] = box[:, 1, :, :] + box[:, 3, :, :]/ 2

    return xyxy

def post_process(input_data, anchors):
    boxes, scores, classes_conf = [], [], []
    input_data = [_in.reshape([len(anchors[0]),-1]+list(_in.shape[-2:])) for _in in input_data]
    for i in range(len(input_data)):
        boxes.append(box_process(input_data[i][:,:4,:,:], anchors[i]))
        scores.append(input_data[i][:,4:5,:,:])
        classes_conf.append(input_data[i][:,5:,:,:])

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print("Detect: %s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# ==========================================
# 4. 主流程: RKNN 模拟器推理验证
# ==========================================
def main():
    rknn = RKNN(verbose=False)

    print('--> Config model')
    # 按照官方 convert.py 的做法配置归一化参数
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=TARGET_PLATFORM)

    print('--> Loading ONNX model')
    rknn.load_onnx(model=ONNX_MODEL)

    print(f'--> Building model (Quantization: {DO_QUANT})')
    rknn.build(do_quantization=DO_QUANT, dataset=DATASET_TXT)

    print('--> Init runtime on PC Simulator')
    # 【核心】：target=None 启动 CPU 模拟器
    ret = rknn.init_runtime(target=None)
    if ret != 0:
        print('Init runtime failed!')
        exit(ret)

    print('--> Loading and preprocessing test image')
    img_src = cv2.imread(TEST_IMG)
    if img_src is None:
        print(f"ERROR: Cannot read image {TEST_IMG}")
        exit(1)
    
    # 简易前处理：直接 Resize (为了方便在独立脚本里运行，我们忽略保持宽高比的 letter_box)
    img = cv2.resize(img_src, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print('--> Running inference on simulator (This might take a few seconds)...')
    # RKNN 推理，输入要求是一张图片组成的 list
    outputs = rknn.inference(inputs=[img_rgb])
    print('--> Inference done! Executing Post-Process...')

    # 后处理解码
    boxes, classes, scores = post_process(outputs, ANCHORS)

    if boxes is not None:
        # 在我们 Resize 后的图片上画框
        img_draw = img.copy()
        draw(img_draw, boxes, scores, classes)
        
        # 保存结果图片
        save_path = 'simulator_result.jpg'
        cv2.imwrite(save_path, img_draw)
        print(f'\n[SUCCESS] Detection result saved to: {save_path}')
    else:
        print('\n[INFO] No objects detected.')

    rknn.release()

if __name__ == '__main__':
    main()
