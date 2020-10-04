# yolov5-tensorrt
A tensorrt implementation of yolov5: https://github.com/ultralytics/yolov5

# requirement
Please use torch>=1.6.0 + onnx>=1.6.0 + TRT 7.1+ (fix upsample issue) to run the sample code  
onnx-simplifier-0.2.16

# The code
Add newly implemented upsample to get this working with current combination of onnx and tensorrt.  
0. prepare above mentioned environment.
1. git clone && git submodule update --init
2. download weights file (use yolov5/models/export.py)
3. python main.py to run the benchmark
4. Generally, for image of size 640*640, using batchsize=1, the speedup is 4x on V100.

# Updates
- 20201004 update to track yolov5 - v3.0 release. download model file from official websites please.

# TODO
- [x] NMS support
- [ ] dynamic shape or dynamic batchsize support (**won't implement soon because onnx-simplifier only supports fixed shape**)
- [ ] FP16 numerical issue and performance investigation
- [ ] Benchmark