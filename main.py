import sys
sys.path.append('yolov5')

from models.common import *
from utils.torch_utils import *
from yolo import *
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

device = torch_utils.select_device('0')
weights = 'yolov5s.pt'
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def GiB(val):
    return val * 1 << 30


def load_model():
    # Load model
    model = Model('yolov5s.yaml').to(device)
    ckpt = torch.load(weights, map_location=device)
    ckpt['model'] = \
                {k: v for k, v in ckpt['model'].state_dict().items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return model


def export_onnx(model, batch_size):
    img = torch.zeros((batch_size, 3, 640, 640)).to(device)
    torch.onnx.export(model, (img), 'yolov5_{}.onnx'.format(batch_size), 
           input_names=["data"], output_names=["model/22"], verbose=True, opset_version=10, operator_export_type=torch.onnx.OperatorExportTypes.ONNX
    )


def build_engine(onnx_path, using_half):
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
            img = torch.zeros((batch_size, 3, 640, 640)).numpy()
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
        img = torch.zeros((batch_size, 3, 640, 640)).to(device)
        if using_half:
            img = img.half()
        start_t = time.time()
        _ = model(img)
        end_t = time.time()
        _[0].cpu()
        [i.cpu() for i in _[1]]
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


if __name__ == '__main__':
    batch_size = 1
    using_half = False
    model = load_model()
    export_onnx(model, batch_size)

    profile_trt(build_engine('yolov5_{}.onnx'.format(batch_size), using_half), batch_size)
    if using_half:
        model.half()
    profile_torch(model, using_half, batch_size)
    