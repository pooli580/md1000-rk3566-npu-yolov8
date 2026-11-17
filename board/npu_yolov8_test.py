"""MD1000 RK3566 NPU YOLOv8 end-to-end inference test v2 (numpy dfl, no torch)."""
import os
import sys
import time
import threading

HOME = os.environ.get('RKNN_HOME', os.path.expanduser('~/rknn-yolov8'))
sys.path.insert(0, os.path.join(HOME, 'rknn_model_zoo'))
sys.path.insert(0, os.path.join(HOME, 'rknn_model_zoo/examples/yolov8/python'))

import cv2
import numpy as np

import rknnlite.api.rknn_runtime as runtime
from rknnlite.api import RKNNLite
runtime.RKNNRuntime._get_rknn_api_lib_path = lambda self: os.path.join(HOME, 'librknnrt.so')

from py_utils.coco_utils import COCO_test_helper

co_helper = COCO_test_helper()
IMG_SIZE = (640, 640)

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HOME, 'yolov8n-rk3566-fp.rknn')
IMG_PATH = os.path.join(HOME, 'yolov8n_bus_result.jpg')
OUT_PATH = MODEL.replace('.rknn', '_result.jpg')

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
           'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
           'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
           'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
           'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
           'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
           'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
           'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
           'toothbrush')


def dfl(position):
    """numpy equivalent of model zoo dfl (avoids torch dependency on board)."""
    x = np.asarray(position)
    n, c, h, w = x.shape
    p_num, mc = 4, c // 4
    y = x.reshape(n, p_num, mc, h, w)
    y = y - y.max(axis=2, keepdims=True)
    e = np.exp(y)
    y = e / e.sum(axis=2, keepdims=True)
    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y * acc).sum(axis=2)


def box_process(position):
    position = np.asarray(position)
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)
    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    return np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)


def nms_boxes(boxes, scores):
    x1, y1, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x1[i] + w[i], x1[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y1[i] + h[i], y1[order[1:]] + h[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[np.where(iou <= 0.45)[0] + 1]
    return np.array(keep, dtype=np.int32)


def post_process(input_data, conf_thres=0.25):
    boxes, scores, classes_conf = [], [], []
    n_branch = 3
    per = len(input_data) // n_branch
    for i in range(n_branch):
        boxes.append(box_process(input_data[per * i]))
        classes_conf.append(np.asarray(input_data[per * i + 1]))
        scores.append(np.ones_like(classes_conf[i][:, :1, :, :]))

    def flatten(_in):
        ch = _in.shape[1]
        return _in.transpose(0, 2, 3, 1).reshape(-1, ch)

    boxes, classes_conf, scores = map(lambda a: [flatten(x) for x in a], (boxes, classes_conf, scores))
    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    box_xyxy = np.concatenate([boxes[:, :2] - boxes[:, 2:4] / 2, boxes[:, :2] + boxes[:, 2:4] / 2], axis=1)
    max_scores = classes_conf.max(axis=1)
    class_ids = classes_conf.argmax(axis=1)
    mask = max_scores > conf_thres
    fb, fc, fids = box_xyxy[mask], max_scores[mask], class_ids[mask]
    dets = []
    for cid in np.unique(fids):
        m = fids == cid
        keep = nms_boxes(fb[m], fc[m])
        for k in keep:
            b = fb[m][k]
            dets.append(([b[0], b[1], b[2] - b[0], b[3] - b[1]], float(fc[m][k]), int(cid)))
    return dets


rknn = RKNNLite(verbose=False)
print('load_rknn:', rknn.load_rknn(MODEL), flush=True)
print('init_runtime:', rknn.init_runtime(), flush=True)

img_src = cv2.imread(IMG_PATH)
img = co_helper.letter_box(im=img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0, 0, 0))
img_data = [np.expand_dims(img[:, :, ::-1], 0).astype(np.uint8)]

outputs = rknn.inference(inputs=img_data)

# sample NPU load while inferring
load_samples = []
stop = threading.Event()
def sampler():
    while not stop.is_set():
        try:
            with open('/sys/kernel/debug/rknpu/load') as f:
                load_samples.append(f.read().strip())
        except Exception:
            pass
        time.sleep(0.3)
th = threading.Thread(target=sampler, daemon=True)
th.start()

N = 30
t0 = time.time()
for _ in range(N):
    outputs = rknn.inference(inputs=img_data)
dt = (time.time() - t0) / N * 1000
stop.set(); th.join()
print(f'\nNPU inference x{N}: avg {dt:.1f} ms/frame ({1000/dt:.1f} FPS)', flush=True)
print('NPU load samples:', sorted(set(load_samples))[-3:] if load_samples else 'n/a')

dets = post_process(outputs)
print(f'detections: {len(dets)}')
for (x, y, w, h), score, cid in dets[:10]:
    print(f'  {CLASSES[cid]:14s} score={score:.2f} box=({int(x)},{int(y)},{int(w)}x{int(h)})')

img_out = img.copy()
for (x, y, w, h), score, cid in dets:
    p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
    cv2.rectangle(img_out, p1, p2, (255, 0, 0), 2)
    cv2.putText(img_out, f'{CLASSES[cid]} {score:.2f}', (int(x), max(int(y) - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.imwrite(OUT_PATH, img_out)
print('saved:', OUT_PATH)
rknn.release()
print('TEST PASS')
