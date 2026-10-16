"""Download official pretrained CULane ResNet18 weights and export CPU ONNX.

No training is performed. Architecture and inference layout follow
https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2 (MIT, Zequn Qin).
Requires optional torch, torchvision and onnx packages for export only.
"""

import argparse
import hashlib
import os
from pathlib import Path
import urllib.request

CHECKPOINT_URL = ("https://drive.usercontent.google.com/download?"
                  "id=1oEjJraFr-3lxhX_OXduAGFWalWa6Xh3W&export=download&confirm=t")
CHECKPOINT_SHA256 = '956616371ee758551c455d28c1c8a9732b39d7fbd85bc70b9869424a5121f967'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('outputs/culane_res18.pth'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--fp16', action='store_true', help='Export GPU-only FP16 weights with float32 input/output')
    parser.add_argument('--force', action='store_true', help='Replace an existing exported model')
    args = parser.parse_args()
    if args.output is None:
        args.output = Path('models/ufldv2_culane_res18_fp16.onnx' if args.fp16
                           else 'models/ufldv2_culane_res18.onnx')
    if args.output.exists() and not args.force:
        parser.error(f'{args.output} already exists; use --force to replace it')
    try:
        import torch
        import torchvision
        import onnx
    except ImportError as exc:
        parser.error(f'Export requires torch, torchvision and onnx: {exc}')

    if not args.checkpoint.exists():
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        partial = args.checkpoint.with_suffix('.pth.part')
        print('Downloading official CULane checkpoint (about 825 MB)...', flush=True)
        with urllib.request.urlopen(CHECKPOINT_URL, timeout=60) as response, partial.open('wb') as target:
            if 'text/html' in response.headers.get('Content-Type', ''):
                raise RuntimeError('The model host returned a web page instead of model weights')
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        os.replace(partial, args.checkpoint)

    if sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError('Checkpoint checksum does not match the verified official CULane weights')

    from collections import OrderedDict

    class CulaneNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            backbone = torchvision.models.resnet18(weights=None)
            self.model = torch.nn.Sequential(OrderedDict(
                (name, layer) for name, layer in backbone.named_children()
                if name not in {'avgpool', 'fc'}))
            self.pool = torch.nn.Conv2d(512, 8, 1)
            self.cls = torch.nn.Sequential(torch.nn.LayerNorm(4000),
                torch.nn.Linear(4000, 2048), torch.nn.ReLU(), torch.nn.Linear(2048, 91224))

        def forward(self, image):
            flat = self.cls(self.pool(self.model(image)).flatten(1))
            row, col, row_exists, col_exists = torch.split(flat, [57600, 32400, 576, 648], dim=1)
            return (row.reshape(-1, 200, 72, 4), col.reshape(-1, 100, 81, 4),
                    row_exists.reshape(-1, 2, 72, 4), col_exists.reshape(-1, 2, 81, 4))

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    network = CulaneNetwork().eval()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    weights = checkpoint.get('model', checkpoint)
    network.load_state_dict({name.removeprefix('module.'): value for name, value in weights.items()}, strict=True)
    del checkpoint, weights
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_model = args.output.with_suffix('.onnx.part')
    with torch.no_grad():
        torch.onnx.export(network, torch.zeros(1, 3, 320, 1600), str(partial_model),
            input_names=['input'], output_names=['loc_row', 'loc_col', 'exist_row', 'exist_col'],
            opset_version=17, dynamo=False)
    if args.fp16:
        from onnxconverter_common import float16
        converted = float16.convert_float_to_float16(onnx.load(str(partial_model)), keep_io_types=True)
        onnx.save(converted, str(partial_model))
    onnx.checker.check_model(str(partial_model))
    os.replace(partial_model, args.output)
    digest = sha256(args.output)
    print(f'Exported {args.output} ({args.output.stat().st_size} bytes)\nSHA256: {digest}')


if __name__ == '__main__':
    main()
