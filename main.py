import sys
sys.path.append('yolov5')

from models.common import *
from utils.torch_utils import *
from yolo import *


def load_model():
    device = torch_utils.select_device('0')
    weights = 'yolov5s.pt'
    # Load model
    #model = torch.load(weights, map_location=device)['model']
    model = Model('yolov5s.yaml').to(device)
    ckpt = torch.load(weights, map_location=device)
    ckpt['model'] = \
                {k: v for k, v in ckpt['model'].state_dict().items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(ckpt['model'])#, strict=False)
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.eval()

    img = torch.zeros((32, 3, 640, 640)).to(device)
    #_ = model(img)
    #m = ckpt['model']
    #names = m.names if hasattr(m, 'names') else m.modules.names
    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    torch.onnx.export(model, (img), 'yolov5.onnx', 
           input_names=["data"], output_names=["model/22"], verbose=True, opset_version=10, operator_export_type=torch.onnx.OperatorExportTypes.ONNX
    )

    print(model(img))


if __name__ == '__main__':
    load_model()