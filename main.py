import sys
sys.path.append('yolov5')

from models.common import *
from utils.torch_utils import *
from utils.datasets import *
from yolo import *
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import copy
import numpy as np
import os
from onnxsim import simplify
import onnx
import struct
import yaml

device = torch_utils.select_device('0')
weights = 'yolov5s.pt'
model_config = 'yolov5s.yaml'
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
image_loader = LoadImages('yolov5/inference/images', img_size=640)
image_loader.__iter__()
_, input_img, _, _ = image_loader.__next__()
input_img = input_img.astype(np.float)
input_img /= 255.0
input_img = np.expand_dims(input_img, axis=0)
with open(model_config) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    num_classes = cfg['nc']

# nms config
conf_thres = 0.4
iou_thres = 0.5
max_det = 300
# nms GPU
topK = 512 # max supported is 4096, if conf_thres is low, such as 0.001, use larger number.
keepTopK = max_det


def GiB(val):
    return val * 1 << 30


#  different from yolov5/utils/non_max_suppression, xywh2xyxy(x[:, :4]) is no longer needed (contained in Detect())
def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = x[:, :4] #xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def load_model():
    # Load model
    model = Model(model_config).to(device)
    ckpt = torch.load(weights, map_location=device)
    ckpt['model'] = \
                {k: v for k, v in ckpt['model'].state_dict().items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return model


def export_onnx(model, batch_size):
    _,_,x,y = input_img.shape
    img = torch.zeros((batch_size, 3, x, y)).to(device)
    torch.onnx.export(model, (img), 'yolov5_{}.onnx'.format(batch_size), 
           input_names=["data"], output_names=["prediction"], verbose=True, opset_version=10, operator_export_type=torch.onnx.OperatorExportTypes.ONNX
    )


def simplify_onnx(onnx_path):
    model = onnx.load(onnx_path)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_path)


def build_engine(onnx_path, using_half):
    engine_file = onnx_path.replace(".onnx", ".engine")
    if os.path.exists(engine_file):
        with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = 1 # always 1 for explicit batch
        config = builder.create_builder_config()
        config.max_workspace_size = GiB(1)
        if using_half:
            config.set_flag(trt.BuilderFlag.FP16)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        
        previous_output = network.get_output(0)
        network.unmark_output(previous_output)

        # slice boxes, obj_score, class_scores
        strides = trt.Dims([1,1,1])
        starts = trt.Dims([0,0,0])
        bs, num_boxes, _ = previous_output.shape
        shapes = trt.Dims([bs, num_boxes, 4])
        boxes = network.add_slice(previous_output, starts, shapes, strides)
        starts[2] = 4
        shapes[2] = 1
        obj_score = network.add_slice(previous_output, starts, shapes, strides)
        starts[2] = 5
        shapes[2] = num_classes
        scores = network.add_slice(previous_output, starts, shapes, strides)

        indices = network.add_constant(trt.Dims([num_classes]), trt.Weights(np.zeros(num_classes, np.int32)))
        gather_layer = network.add_gather(obj_score.get_output(0), indices.get_output(0), 2)

        # scores = obj_score * class_scores => [bs, num_boxes, nc]
        updated_scores = network.add_elementwise(gather_layer.get_output(0), scores.get_output(0), trt.ElementWiseOperation.PROD)

        # reshape box to [bs, num_boxes, 1, 4]
        reshaped_boxes = network.add_shuffle(boxes.get_output(0))
        reshaped_boxes.reshape_dims = trt.Dims([0,0,1,4])

        # add batchedNMSPlugin, inputs:[boxes:(bs, num, 1, 4), scores:(bs, num, 1)]
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        registry = trt.get_plugin_registry()
        assert(registry)
        creator = registry.get_plugin_creator("BatchedNMS_TRT", "1")
        assert(creator)
        fc = []
        fc.append(trt.PluginField("shareLocation", np.array([1], dtype=np.int), trt.PluginFieldType.INT32))
        fc.append(trt.PluginField("backgroundLabelId", np.array([-1], dtype=np.int), trt.PluginFieldType.INT32))
        fc.append(trt.PluginField("numClasses", np.array([num_classes], dtype=np.int), trt.PluginFieldType.INT32))
        fc.append(trt.PluginField("topK", np.array([topK], dtype=np.int), trt.PluginFieldType.INT32))
        fc.append(trt.PluginField("keepTopK", np.array([keepTopK], dtype=np.int), trt.PluginFieldType.INT32))
        fc.append(trt.PluginField("scoreThreshold", np.array([conf_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32))
        fc.append(trt.PluginField("iouThreshold", np.array([iou_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32))
        fc.append(trt.PluginField("isNormalized", np.array([0], dtype=np.int), trt.PluginFieldType.INT32))
        fc.append(trt.PluginField("clipBoxes", np.array([0], dtype=np.int), trt.PluginFieldType.INT32))
        
        fc = trt.PluginFieldCollection(fc) 
        nms_layer = creator.create_plugin("nms_layer", fc)

        layer = network.add_plugin_v2([reshaped_boxes.get_output(0), updated_scores.get_output(0)], nms_layer)
        layer.get_output(0).name = "num_detections"
        layer.get_output(1).name = "nmsed_boxes"
        layer.get_output(2).name = "nmsed_scores"
        layer.get_output(3).name = "nmsed_classes"
        for i in range(4):
            network.mark_output(layer.get_output(i))

        return builder.build_engine(network, config)


def allocate_buffers(engine, is_explicit_batch=False, dynamic_shapes=[]):
    inputs = []
    outputs = []
    bindings = []

    class HostDeviceMem(object):
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

        def __str__(self):
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

        def __repr__(self):
            return self.__str__()

    for binding in engine:
        dims = engine.get_binding_shape(binding)
        print(dims)
        if dims[0] == -1:
            assert(len(dynamic_shapes) > 0)
            dims[0] = dynamic_shapes[0]
        size = trt.volume(dims) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings


def profile_trt(engine, batch_size, num_warmups=10, num_iters=100):
    assert(engine is not None)

    yolo_inputs, yolo_outputs, yolo_bindings = allocate_buffers(engine, True)
    
    stream = cuda.Stream()    
    with engine.create_execution_context() as context:
        
        total_duration = 0.
        total_compute_duration = 0.
        total_pre_duration = 0.
        total_post_duration = 0.
        for iteration in range(num_iters):
            pre_t = time.time()
            # set host data
            img = torch.from_numpy(input_img).float().numpy()
            yolo_inputs[0].host = img
            [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in yolo_inputs]
            stream.synchronize()
            start_t = time.time()
            context.execute_async_v2(bindings=yolo_bindings, stream_handle=stream.handle)
            stream.synchronize()
            end_t = time.time()
            [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in yolo_outputs]
            stream.synchronize()
            post_t = time.time()

            duration = post_t - pre_t
            compute_duration = end_t - start_t
            pre_duration = start_t - pre_t
            post_duration = post_t - end_t
            if iteration >= num_warmups:
                total_duration += duration
                total_compute_duration += compute_duration
                total_post_duration += post_duration
                total_pre_duration += pre_duration
        
        print("avg GPU time: {}".format(total_duration/(num_iters - num_warmups)))
        print("avg GPU compute time: {}".format(total_compute_duration/(num_iters - num_warmups)))
        print("avg pre time: {}".format(total_pre_duration/(num_iters - num_warmups)))
        print("avg post time: {}".format(total_post_duration/(num_iters - num_warmups)))
        
        num_det = int(yolo_outputs[0].host)
        boxes = np.array(yolo_outputs[1].host.reshape(1, -1, 4))[:, 0:num_det, :]
        scores = np.array(yolo_outputs[2].host.reshape(1, -1, 1))[:, 0:num_det, :]
        classes = np.array(yolo_outputs[3].host.reshape(1, -1, 1))[:, 0:num_det, :]
        
        return [np.concatenate([boxes, scores, classes], -1)]


def profile_torch(model, using_half, batch_size, num_warmups=10, num_iters=100):
    total_duration = 0.
    total_compute_duration = 0.
    total_pre_duration = 0.
    total_post_duration = 0.
    if using_half:
        model.half()
    for iteration in range(num_iters):
        pre_t = time.time()
        # set host data
        img = torch.from_numpy(input_img).float().to(device)
        if using_half:
            img = img.half()
        start_t = time.time()
        _ = model(img)
        output = non_max_suppression(_[0], conf_thres, iou_thres)
        end_t = time.time()
        [i.cpu() for i in _]
        post_t = time.time()

        duration = post_t - pre_t
        compute_duration = end_t - start_t
        pre_duration = start_t - pre_t
        post_duration = post_t - end_t
        if iteration >= num_warmups:
            total_duration += duration
            total_compute_duration += compute_duration
            total_post_duration += post_duration
            total_pre_duration += pre_duration
    
    print("avg GPU time: {}".format(total_duration/(num_iters - num_warmups)))
    print("avg GPU compute time: {}".format(total_compute_duration/(num_iters - num_warmups)))
    print("avg pre time: {}".format(total_pre_duration/(num_iters - num_warmups)))
    print("avg post time: {}".format(total_post_duration/(num_iters - num_warmups)))

    return [output[0].cpu().numpy()]


if __name__ == '__main__':
    batch_size = 1
    using_half = False
    onnx_path = 'yolov5_{}.onnx'.format(batch_size)
    
    with torch.no_grad():
        model = load_model()
        export_onnx(model, batch_size)
        simplify_onnx(onnx_path)

        trt_result = profile_trt(build_engine(onnx_path, using_half), batch_size, 10, 100)
        if using_half:
            model.half()
        torch_result = profile_torch(model, using_half, batch_size, 10, 100)
        
        print(trt_result)
        print(torch_result)

    
