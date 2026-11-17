"""Parallel-client test: 2 processes inferring concurrently.
If combined FPS ~= 2x single, the idle gaps are per-client overhead (pipeline stalls);
if combined ~= 1x, NPU hardware queue is the real bottleneck."""
import os, sys, time
HOME = os.environ.get('RKNN_HOME', os.path.expanduser('~/rknn-yolov8'))
sys.path.insert(0, os.path.join(HOME, 'rknn_model_zoo'))
sys.path.insert(0, os.path.join(HOME, 'rknn_model_zoo/examples/yolov8/python'))
import numpy as np, cv2
import rknnlite.api.rknn_runtime as runtime
from rknnlite.api import RKNNLite
runtime.RKNNRuntime._get_rknn_api_lib_path = lambda self: os.path.join(HOME, 'librknnrt.so')
from py_utils.coco_utils import COCO_test_helper

co = COCO_test_helper()
rk = RKNNLite(verbose=False)
MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HOME, 'yolov8n-rk3566-fp.rknn')
rk.load_rknn(MODEL)
rk.init_runtime()
img = co.letter_box(im=cv2.imread(os.path.join(HOME, 'yolov8n_bus_result.jpg')).copy(),
                    new_shape=(416, 416), pad_color=(0, 0, 0))
x = [np.expand_dims(img[:, :, ::-1], 0).astype(np.uint8)]
rk.inference(inputs=x)

N = 100
t0 = time.time()
for _ in range(N):
    rk.inference(inputs=x)
dt = time.time() - t0
print(f'CLIENT: {N} frames in {dt:.1f}s = {N/dt:.2f} FPS ({dt/N*1000:.1f} ms/frame)')
rk.release()
