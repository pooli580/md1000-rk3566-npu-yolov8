"""NPU FPS benchmark: pure inference throughput + NPU load sampling as root."""
import os
import sys
import time
import threading
import subprocess

HOME = os.environ.get('RKNN_HOME', os.path.expanduser('~/rknn-yolov8'))
sys.path.insert(0, os.path.join(HOME, 'rknn_model_zoo'))
sys.path.insert(0, os.path.join(HOME, 'rknn_model_zoo/examples/yolov8/python'))
import numpy as np
import cv2
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
rk.inference(inputs=x)  # warmup

load_samples = []
stop = threading.Event()

def sampler():
    while not stop.is_set():
        try:
            out = subprocess.run(
                ['sudo', '-S', 'cat', '/sys/kernel/debug/rknpu/load'],
                input=os.environ.get('SUDO_PASS', '') + '\n', capture_output=True, text=True, timeout=5)
            for line in out.stdout.splitlines():
                if 'load' in line.lower() or '%' in line:
                    load_samples.append(line.strip())
        except Exception:
            pass
        time.sleep(0.2)

th = threading.Thread(target=sampler, daemon=True)
th.start()
time.sleep(0.5)

N = 200
t0 = time.time()
for _ in range(N):
    rk.inference(inputs=x)
total = time.time() - t0
stop.set(); th.join()

fps = N / total
print(f'\nPURE NPU INFERENCE: {N} frames in {total:.2f}s')
print(f'  avg latency: {total/N*1000:.1f} ms/frame')
print(f'  throughput : {fps:.2f} FPS')
if load_samples:
    import re
    vals = []
    for s in load_samples:
        m = re.findall(r'(\d+)%', s)
        vals += [int(v) for v in m]
    if vals:
        print(f'  NPU load during run: max {max(vals)}%, avg {sum(vals)/len(vals):.0f}%')
rk.release()
