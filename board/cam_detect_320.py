#!/usr/bin/env python3
"""Live USB camera YOLOv8 on MD1000 NPU — pipelined capture/inference, 320 int8."""
import os
import sys
import time
import threading

import cv2
import numpy as np

HOME = os.environ.get('RKNN_HOME', os.path.expanduser('~/rknn-yolov8'))
import rknnlite.api.rknn_runtime as runtime
from rknnlite.api import RKNNLite
runtime.RKNNRuntime._get_rknn_api_lib_path = lambda self: os.path.join(HOME, 'librknnrt.so')

IMG_SIZE = (320, 320)
MODEL = os.path.join(HOME, 'yolov8n-320-int8-tk16.rknn')
OUT_PATH = os.path.join(HOME, 'cam_detect.jpg')
N_FRAMES = 200
CONF = 0.35

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
    x = np.asarray(position)
    n, c, h, w = x.shape
    p_num, mc = 4, c // 4
    y = x.reshape(n, p_num, mc, h, w)
    y = y - y.max(axis=2, keepdims=True)
    e = np.exp(y)
    y = e / e.sum(axis=2, keepdims=True)
    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y * acc).sum(axis=2)


def box_process(position, cls_map=None, conf_th=0.05):
    position = np.asarray(position)
    grid_h, grid_w = position.shape[2:4]
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)
    if cls_map is not None:
        cm = np.asarray(cls_map)[0].max(axis=0)  # (H, W) max over classes
        idx_h, idx_w = np.where(cm > conf_th)
        K = len(idx_h)
        if K == 0:
            return np.zeros((1, 4, 1, 0), np.float32), None
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col[idx_h, idx_w].reshape(1, 1, 1, K)
        row = row[idx_h, idx_w].reshape(1, 1, 1, K)
        grid = np.concatenate((col, row), axis=1)
        pos_sel = np.ascontiguousarray(position[:, :, idx_h, idx_w].reshape(1, position.shape[1], 1, K))
        position = dfl(pos_sel)
        cls_sel = None  # handled by caller via idx
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        boxes = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
        return boxes, (idx_h, idx_w)
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    return np.concatenate((box_xy * stride, box_xy2 * stride), axis=1), None


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
        yy2 = np.minimum(y1[i] + h[i], x1[order[1:]] + h[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[np.where(iou <= 0.45)[0] + 1]
    return np.array(keep, dtype=np.int32)


def post_process(input_data, conf_thres=CONF):
    boxes, scores, classes_conf = [], [], []
    per = len(input_data) // 3
    for i in range(3):
        b, sel = box_process(input_data[per * i], input_data[per * i + 1], conf_thres)
        nb = b.transpose(0, 2, 3, 1).reshape(-1, 4).shape[0]
        boxes.append(b.transpose(0, 2, 3, 1).reshape(-1, 4))
        if isinstance(sel, tuple):
            idx_h, idx_w = sel
            cls_sel = np.asarray(input_data[per * i + 1])[:, :, idx_h, idx_w]  # (1,80,K)
            classes_conf.append(cls_sel.transpose(0, 2, 1).reshape(-1, 80))
        elif nb == 0:
            classes_conf.append(np.zeros((0, 80), np.float32))
        else:
            classes_conf.append(np.asarray(input_data[per * i + 1]).transpose(0, 2, 3, 1).reshape(-1, 80))
        scores.append(np.ones((boxes[-1].shape[0], 1), np.float32))
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


def letterbox(im, new_shape):
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(h * r), int(w * r)
    resized = cv2.resize(im, (nw, nh))
    top = (new_shape[0] - nh) // 2
    bottom = new_shape[0] - nh - top
    left = (new_shape[1] - nw) // 2
    right = new_shape[1] - nw - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded, r, left, top


class CameraThread(threading.Thread):
    """Grab + letterbox + RGB convert in background; hand ready-to-infer buffers."""

    def __init__(self, dev='/dev/video1'):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.frame = None
        self.ready = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            ret, f = self.cap.read()
            if ret and f is not None:
                lb, r, left, top = letterbox(f, IMG_SIZE)
                rgb = np.ascontiguousarray(lb[:, :, ::-1])
                with self.lock:
                    self.frame = f
                    self.ready = (rgb, r, left, top)

    def get(self):
        with self.lock:
            if self.ready is None:
                return None, None
            return self.frame, self.ready

    def stop(self):
        self.running = False
        self.cap.release()


rknn = RKNNLite(verbose=False)
print('load_rknn:', rknn.load_rknn(MODEL), flush=True)
print('init_runtime:', rknn.init_runtime(), flush=True)

cam = CameraThread('/dev/video1')
cam.start()
# wait first frame
for _ in range(50):
    if cam.get() is not None:
        break
    time.sleep(0.1)
if cam.get() is None:
    print('CAMERA FAILED')
    sys.exit(1)
print('camera thread running', flush=True)

inf_ms_sum, loop_ms_sum, n_ok, drop = 0.0, 0.0, 0, 0
last_dets, last_frame = [], None
t_loop = time.time()
for i in range(N_FRAMES):
    frame, ready = cam.get()
    if frame is None:
        time.sleep(0.005)
        drop += 1
        continue
    rgb, r, left, top = ready
    t0 = time.time()
    outputs = rknn.inference(inputs=[np.expand_dims(rgb, 0)])
    dt_inf = time.time() - t0
    inf_ms_sum += dt_inf * 1000
    dets = post_process(outputs)
    mapped = []
    for (x, y, w, h), score, cid in dets:
        mapped.append((( (x - left) / r, (y - top) / r, w / r, h / r), score, cid))
    last_dets = mapped
    if i >= N_FRAMES - 3:
        last_frame = frame.copy()
    loop_ms_sum += (time.time() - t_loop) * 1000
    t_loop = time.time()
    n_ok += 1

cam.stop()

print(f'\npipelined loop x{n_ok} (drop {drop}): end-to-end avg {loop_ms_sum/n_ok:.1f} ms/frame ({1000*n_ok/loop_ms_sum:.1f} FPS)', flush=True)
print(f'pure inference: avg {inf_ms_sum/n_ok:.1f} ms/frame ({1000/inf_ms_sum*n_ok:.1f} FPS)', flush=True)
print(f'detections in last frame: {len(last_dets)}')
img_out = last_frame
for (x, y, w, h), score, cid in last_dets:
    p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
    cv2.rectangle(img_out, p1, p2, (255, 0, 0), 2)
    cv2.putText(img_out, f'{CLASSES[cid]} {score:.2f}', (int(x), max(int(y) - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.imwrite(OUT_PATH, img_out)
print('saved:', OUT_PATH, flush=True)
for (x, y, w, h), score, cid in last_dets[:10]:
    print(f'  {CLASSES[cid]:14s} score={score:.2f} box=({int(x)},{int(y)},{int(w)}x{int(h)})')
rknn.release()
print('CAM 320 PIPELINE TEST PASS', flush=True)
