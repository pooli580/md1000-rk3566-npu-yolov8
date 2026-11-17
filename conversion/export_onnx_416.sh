#!/bin/bash
set -x
cd /home/<user>/ultralytics_yolov8
sed -i "s|if self.export and self.format == 'rknn':|if self.export and self.format in ('rknn', 'onnx'):|" ultralytics/nn/modules/head.py
grep -n "self.format in" ultralytics/nn/modules/head.py | head -2
sed -i "s|^format: rknn.*|format: onnx|" ultralytics/cfg/default.yaml
grep -E '^(model|imgsz|format):' ultralytics/cfg/default.yaml
export PYTHONPATH=/home/<user>/ultralytics_yolov8
export PYTHONNOUSERSITE=1
/home/<user>/miniconda3/envs/yolov8/bin/python ./ultralytics/engine/exporter.py > /home/<user>/exp416onnx.log 2>&1
tail -6 /home/<user>/exp416onnx.log
ls -la /home/<user>/yolov8n.onnx
