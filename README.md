# yolov5-tensorrt
A tensorrt implementation of yolov5: https://github.com/ultralytics/yolov5

# requirement
Please use torch==1.4.0 + onnx==1.6.0 + TRT 7.0+ to run the sample code  
onnx-simplifier-0.2.9

# The code
Add newly implemented upsample to get this working with current combination of onnx and tensorrt.  
1. download weights file
2. modify yolov5*.yaml, replace nn.Upsample with Upsample
3. python main.py to run the benchmark
4. Generally, for image of size 640*640, using batchsize=1, the speedup is 4x on V100.

# TODO
- [x] NMS support
- [ ] dynamic shape or dynamic batchsize support (**won't implement soon because onnx-simplifier only supports fixed shape**)
- [ ] FP16 numerical issue and performance investigation
- [ ] Benchmark
- [ ] Standalone infer script