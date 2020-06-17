import sys
sys.path.append('yolov5')

from models.common import *
from utils.torch_utils import *
from yolo import *
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

device = torch_utils.select_device('0')
weights = 'yolov5s.pt'


def load_model():
    # Load model
    #model = torch.load(weights, map_location=device)['model']
    model = Model('yolov5s.yaml').to(device)
    ckpt = torch.load(weights, map_location=device)
    ckpt['model'] = \
                {k: v for k, v in ckpt['model'].state_dict().items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    return model


def export_onnx(model):
    img = torch.zeros((1, 3, 640, 640)).to(device)
    torch.onnx.export(model, (img), 'yolov5.onnx', 
           input_names=["data"], output_names=["model/22"], verbose=True, opset_version=10, operator_export_type=torch.onnx.OperatorExportTypes.ONNX
    )


def build_engine(onnx_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = 1 # always 1 for explicit batch
        config = builder.create_builder_config()
        config.max_workspace_size = GiB(1)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        return builder.build_engine(network, config)


def profile_trt(engine):
    pass


def profile_torch(model):
    pass


if __name__ == '__main__':
    model = load_model()
    export_onnx(model)
    engine = build_engine('yolov5.onnx')
    assert(engine is not None)

    profile_trt(engine)
    profile_torch(model)
    