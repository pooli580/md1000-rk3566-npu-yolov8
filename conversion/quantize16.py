#!/usr/bin/env python3
"""Convert yolov8n.onnx to INT8 with toolkit2 1.6.0 (old model format)."""
import os
from rknn.api import RKNN

WORK = os.environ.get('RKNN_CONV_HOME', os.path.expanduser('~/yolov8n_int8'))

rk = RKNN(verbose=True)

rk.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    quant_img_RGB2BGR=True,
    quantized_algorithm='normal',
    quantized_method='channel',
    target_platform='rk3566',
)

ret = rk.load_onnx(model=f'{WORK}/yolov8n.onnx')
print('load_onnx:', ret)
if ret != 0:
    exit(ret)

ret = rk.build(do_quantization=True, dataset=f'{WORK}/dataset.txt')
print('build:', ret)
if ret != 0:
    exit(ret)

ret = rk.export_rknn(f'{WORK}/yolov8n-rk3566-int8-tk16.rknn')
print('export_rknn:', ret)
if ret != 0:
    exit(ret)

rk.release()
print('TK16 INT8 EXPORTED OK')
