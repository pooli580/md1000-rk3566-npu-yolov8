#!/bin/bash
set -x
cd /home/<user>/ultralytics_yolov8
sed -i 's|^imgsz: 416.*|imgsz: 320|' ultralytics/cfg/default.yaml
grep -E '^(model|imgsz|format|batch):' ultralytics/cfg/default.yaml
export PYTHONPATH=/home/<user>/ultralytics_yolov8
export PYTHONNOUSERSITE=1
/home/<user>/miniconda3/envs/yolov8/bin/python ./ultralytics/engine/exporter.py > /home/<user>/exp320.log 2>&1
tail -3 /home/<user>/exp320.log
/home/<user>/venv_tk16/bin/python -c "
import onnx
m = onnx.load(os.environ.get('RKNN_ONNX', os.path.expanduser('~/yolov8n.onnx')))
dims = [d.dim_value or d.dim_param for d in m.graph.input[0].type.tensor_type.shape.dim]
print('input:', dims)
"
cd /home/<user>/yolov8n_int8
sed 's|yolov8n-416-int8-tk16.rknn|yolov8n-320-int8-tk16.rknn|' quantize16_416.py > quantize16_320.py
/home/<user>/venv_tk16/bin/python quantize16_320.py > tk16_320.log 2>&1
grep -E 'build:|export_rknn:|EXPORTED OK' tk16_320.log | tail -3
ls -la yolov8n-320-int8-tk16.rknn 2>/dev/null
