#!/usr/bin/env python3
"""Convert 416 rknn-optimized yolov8n.onnx to INT8 with toolkit2 1.6.0."""
import os
from rknn.api import RKNN

rk = RKNN(verbose=True)

rk.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    quant_img_RGB2BGR=True,
    quantized_algorithm='normal',
    quantized_method='channel',
    target_platform='rk3566',
)

ret = rk.load_onnx(model=os.environ.get('RKNN_ONNX', os.path.expanduser('~/yolov8n.onnx')))
print('load_onnx:', ret)
if ret != 0:
    exit(ret)

ret = rk.build(do_quantization=True, dataset='/home/<user>/yolov8n_int8/dataset.txt')
print('build:', ret)
if ret != 0:
    exit(ret)

ret = rk.export_rknn('/home/<user>/yolov8n_int8/yolov8n-320-int8-tk16.rknn')
print('export_rknn:', ret)
if ret != 0:
    exit(ret)

rk.release()
print('TK16 416 INT8 EXPORTED OK')
