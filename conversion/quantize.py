#!/usr/bin/env python3
"""Quantize yolov8n.onnx to INT8 RKNN for RK3566 (MD1000).

Matches codex's FP conversion config (rknn_model_zoo yolov8 standard):
mean 0, std 255, float16, w8a8 quantization with coco128 calibration set.
"""
import os
from rknn.api import RKNN

WORK = os.environ.get('RKNN_CONV_HOME', os.path.expanduser('~/yolov8n_int8'))

rknn = RKNN(verbose=True)

rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    quant_img_RGB2BGR=True,
    quantized_dtype='w8a8',
    quantized_algorithm='kl_divergence',
    quantized_method='channel',
    target_platform='rk3566',
    float_dtype='float16',
    optimization_level=3,
    remove_weight=True,
    compress_weight=False,
    output_optimize=False,
    single_core_mode=True,
)

ret = rknn.load_onnx(model=f'{WORK}/yolov8n.onnx')
if ret != 0:
    print('load_onnx failed:', ret)
    exit(ret)

ret = rknn.build(do_quantization=True, dataset=f'{WORK}/dataset.txt')
if ret != 0:
    print('build failed:', ret)
    exit(ret)

ret = rknn.export_rknn(f'{WORK}/yolov8n-rk3566-int8.rknn')
if ret != 0:
    print('export_rknn failed:', ret)
    exit(ret)

rknn.release()
print('INT8 RKNN EXPORTED OK -> yolov8n-rk3566-int8.rknn')
