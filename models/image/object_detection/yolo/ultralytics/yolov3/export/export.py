# Model: https://github.com/ultralytics/yolov3.git

# This script saves the ultralytics/yolov3 model to TorchScript
# so that it can be used in RedisAI (or from other C/C++ runtimes)
#
# For options: python export_trace.py --help
#
# This will generate a file (yolov3-spp-traced.pt) that can
# be used with RedisAI


import argparse
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *


class DarknetExport(Darknet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        x, _ = super().forward(*args, **kwargs)
        return x


def export():
    img_size = opt.img_size
    weights = opt.weights
    device = 'cpu'

    # Initialize model
    model = DarknetExport(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Eval mode
    model.eval()

    # Fuse Conv2d + BatchNorm2d layers
    model.fuse()

    img = torch.rand((1, 3) + (img_size, img_size))

    model(img)

    traced = torch.jit.trace(model, [img])
    torch.jit.save(traced, opt.outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--outfile', type=str, default='yolov3-spp.pt', help='exported file path')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        export()
