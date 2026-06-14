import os
import cv2
import numpy as np
import time
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

# 【注意】板端部署必须使用 rknnlite 库
from rknnlite.api import RKNNLite 

# ==========================================
# 1. 基础参数配置 (RK3588 真机环境)
# ==========================================
RKNN_MODEL = 'dogface.rknn'   # 请确保你已经将转换好的 rknn 模型放到了板子上
CAMERA_ID = 0                 # 摄像头节点，通常是 0 (/dev/video0) 或者特定编号
THREADS = 3                   # RK3588 有 3 个 NPU 核心，开 3 个线程最高效

# ==========================================
# 2. YOLOv5 常量配置
# ==========================================
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)  # (width, height)

ANCHORS = [[[1.25, 1.625], [2.0, 3.75], [4.125, 2.875]],
           [[1.875, 3.8125], [3.875, 2.8125], [3.6875, 7.4375]],
           [[3.625, 2.8125], [4.875, 6.1875], [11.65625, 10.1875]]]

CLASSES = ("dogface")

# ==========================================
# 3. YOLOv5 后处理函数 (保留你原本的代码)
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
        # 视频流实时推理时注释掉 print 避免终端刷屏卡顿
        # print("Detect: %s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# ==========================================
# 4. 核心：NPU 多线程异步推理池
# ==========================================
class rknnPoolExecutor:
    def __init__(self, rknn_model, thread_count, worker_func):
        self.thread_count = thread_count
        self.queue = Queue()
        self.rknnPool = []
        self.worker_func = worker_func
        self.frame_num = 0
        self.pool = ThreadPoolExecutor(max_workers=thread_count)

        print(f"--> 正在初始化 {thread_count} 个 RKNNLite 实例...")
        self._init_rknns(rknn_model)

    def _init_rknns(self, rknn_model):
        # 针对 RK3588 的三个独立 NPU 核心进行分配
        core_masks = [RKNNLite.NPU_CORE_0, RKNNLite.NPU_CORE_1, RKNNLite.NPU_CORE_2]
        
        for i in range(self.thread_count):
            rknn = RKNNLite()
            ret = rknn.load_rknn(rknn_model)
            if ret != 0:
                print(f"Load RKNN model failed in thread {i}!")
                exit(-1)
            
            # 将不同的线程绑定到不同的 NPU 核心上（轮询分配）
            core = core_masks[i % 3]
            ret = rknn.init_runtime(core_mask=core)
            if ret != 0:
                print(f"Init runtime failed in thread {i}!")
                exit(-1)
                
            self.rknnPool.append(rknn)
            print(f"--> 实例 {i} 初始化成功并绑定至 NPU Core {i % 3}")

    def put(self, frame):
        # 提交任务到线程池
        future = self.pool.submit(self.worker_func, self.rknnPool[self.frame_num % self.thread_count], frame)
        self.queue.put(future)
        self.frame_num += 1

    def get(self):
        if self.queue.empty():
            return None, False
        future = self.queue.get()
        return future.result(), True

    def release(self):
        self.pool.shutdown()
        for rknn in self.rknnPool:
            rknn.release()

# ==========================================
# 5. 线程工作函数 (包含预处理、推理、后处理)
# ==========================================
def inference_worker(rknn_instance, frame_src):
    # 1. 预处理
    frame = cv2.resize(frame_src, IMG_SIZE)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 【关键修复】：增加 Batch 维度，将 (640, 640, 3) 变成 (1, 640, 640, 3)
    img_rgb = np.expand_dims(img_rgb, axis=0)

    # 2. NPU 推理
    outputs = rknn_instance.inference(inputs=[img_rgb])

    # 如果推理失败，直接返回原图避免报错
    if outputs is None:
        return frame

    # 3. 后处理解码
    boxes, classes, scores = post_process(outputs, ANCHORS)

    # 4. 画图
    if boxes is not None:
        draw(frame, boxes, scores, classes)

    return frame

# ==========================================
# 6. 主流程：摄像头读取与多线程调度
# ==========================================
def main():
    if not os.path.exists(RKNN_MODEL):
        print(f"错误: 找不到模型文件 {RKNN_MODEL}！请先在 PC 端将 ONNX 转换为 RKNN 并拷入板子。")
        return

    print("\n" + "="*50)
    print(" 开始 RK3588 摄像头多线程实时检测 ")
    print("="*50 + "\n")

    # 初始化推理池
    pool = rknnPoolExecutor(RKNN_MODEL, THREADS, inference_worker)

    # 打开摄像头
    cap = cv2.VideoCapture(CAMERA_ID)
    
    # 可选：降低摄像头分辨率以提升读取帧率，防止 I/O 瓶颈
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"ERROR: 无法打开摄像头 {CAMERA_ID}。")
        pool.release()
        return

    print("--> 摄像头开启成功，开始预填充推理队列...")
    
    # 预填充队列
    for i in range(THREADS + 1):
        ret, frame = cap.read()
        if not ret:
            break
        pool.put(frame)

    frames_processed = 0
    start_time = time.time()

    print("--> 启动流水线，按 'q' 键退出...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("读取摄像头失败！")
            break

        # 放入新帧进行异步推理
        pool.put(frame)
        
        # 获取最老的一帧推理结果
        result_frame, success = pool.get()

        if success:
            frames_processed += 1
            
            # 每 30 帧计算一次 FPS
            if frames_processed % 30 == 0:
                elapsed = time.time() - start_time
                fps = 30 / elapsed
                print(f"当前实时处理 FPS: {fps:.2f}")
                start_time = time.time()

            cv2.imshow("RK3588 YOLOv5 Real-Time NPU", result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 收尾与释放
    cap.release()
    pool.release()
    cv2.destroyAllWindows()
    print("\n--> 运行结束，资源已释放。")

if __name__ == '__main__':
    main()
