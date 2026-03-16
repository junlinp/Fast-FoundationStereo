#!/usr/bin/env python3
"""
Export Fast-FoundationStereo to ONNX with dynamic resolution support.
Allows running at any resolution (height % 32 == 0, width % 32 == 0)
"""

import warnings, argparse, logging, os, sys
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{code_dir}/../')

import torch
from omegaconf import OmegaConf
from core.foundation_stereo import FastFoundationStereo, TrtFeatureRunner, TrtPostRunner
from core.submodule import build_gwc_volume_optimized_pytorch1
import Utils as U


class FoundationStereoOnnx(FastFoundationStereo):
    def __init__(self, args):
        super().__init__(args)

    @torch.no_grad()
    def forward(self, left, right):
        with torch.amp.autocast('cuda', enabled=True, dtype=U.AMP_DTYPE):
            disp = FastFoundationStereo.forward(self, left, right, iters=self.args.valid_iters, test_mode=True, optimize_build_volume=False)
        return disp


def export_dynamic_onnx(args):
    """Export ONNX models with dynamic resolution support."""
    
    torch.autograd.set_grad_enabled(False)
    
    # Load model
    print(fLoading model from {args.model_dir})
    model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
    model.args.max_disp = args.max_disp
    model.args.valid_iters = args.valid_iters
    model.cuda().eval()
    
    feature_runner = TrtFeatureRunner(model)
    post_runner = TrtPostRunner(model)
    feature_runner.cuda().eval()
    post_runner.cuda().eval()
    
    # Export feature_runner with dynamic shapes
    print(fExporting feature_runner.onnx with dynamic resolution...)
    
    left_input = torch.randn(1, 3, 480, 640, device='cuda', dtype=torch.float32)
    right_input = torch.randn(1, 3, 480, 640, device='cuda', dtype=torch.float32)
    
    torch.onnx.export(
        feature_runner,
        (left_input, right_input),
        args.save_path + '/feature_runner.onnx',
        opset_version=17,
        input_names=['left', 'right'],
        output_names=[
            'features_left_04', 'features_left_08', 'features_left_16', 'features_left_32',
            'features_right_04', 'features_right_08', 'features_right_16', 'features_right_32',
            'stem_2x', 'stem_4x'
        ],
        dynamic_axes={
            'left': {0: 'batch', 2: 'height', 3: 'width'},
            'right': {0: 'batch', 2: 'height', 3: 'width'},
            'features_left_04': {0: 'batch', 2: 'height', 3: 'width'},
            'features_left_08': {0: 'batch', 2: 'height', 3: 'width'},
            'features_left_16': {0: 'batch', 2: 'height', 3: 'width'},
            'features_left_32': {0: 'batch', 2: 'height', 3: 'width'},
            'features_right_04': {0: 'batch', 2: 'height', 3: 'width'},
            'features_right_08': {0: 'batch', 2: 'height', 3: 'width'},
            'features_right_16': {0: 'batch', 2: 'height', 3: 'width'},
            'features_right_32': {0: 'batch', 2: 'height', 3: 'width'},
            'stem_2x': {0: 'batch', 2: 'height', 3: 'width'},
            'stem_4x': {0: 'batch', 2: 'height', 3: 'width'},
        },
        do_constant_folding=True,
        verbose=False
    )
    
    print(feature_runner.onnx exported!)
    
    # Export post_runner with dynamic shapes
    print(fExporting post_runner.onnx with dynamic resolution...)
    
    # Get output shapes for post_runner
    with torch.no_grad():
        f_left_04, f_left_08, f_left_16, f_left_32, f_right_04, stem_2x = feature_runner(left_input, right_input)
        gwc_volume = build_gwc_volume_optimized_pytorch1(
            f_left_04.half(), f_right_04.half(), 
            args.max_disp // 4, model.cv_group
        )
    
    torch.onnx.export(
        post_runner,
        (f_left_04.float(), f_left_08.float(), f_left_16.float(), f_left_32.float(),
         f_right_04.float(), stem_2x.float(), gwc_volume.float()),
        args.save_path + '/post_runner.onnx',
        opset_version=17,
        input_names=[
            'features_left_04', 'features_left_08', 'features_left_16', 'features_left_32',
            'features_right_04', 'stem_2x', 'gwc_volume'
        ],
        output_names=['disp'],
        dynamic_axes={
            'features_left_04': {0: 'batch', 2: 'height', 3: 'width'},
            'features_left_08': {0: 'batch', 2: 'height', 3: 'width'},
            'features_left_16': {0: 'batch', 2: 'height', 3: 'width'},
            'features_left_32': {0: 'batch', 2: 'height', 3: 'width'},
            'features_right_04': {0: 'batch', 2: 'height', 3: 'width'},
            'stem_2x': {0: 'batch', 2: 'height', 3: 'width'},
            'gwc_volume': {0: 'batch', 2: 'height', 3: 'width'},
            'disp': {0: 'batch', 1: 'height', 2: 'width'},
        },
        do_constant_folding=True,
        verbose=False
    )
    
    print(post_runner.onnx exported!)
    
    # Save config
    config = {
        'image_size': [args.height, args.width],
        'max_disp': args.max_disp,
        'valid_iters': args.valid_iters,
        'dynamic_resolution': True
    }
    
    import yaml
    with open(args.save_path + '/onnx.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print(fDone! ONNX models saved to {args.save_path})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export Fast-FoundationStereo to ONNX with dynamic resolution')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save ONNX models')
    parser.add_argument('--height', type=int, default=480, help='Default height')
    parser.add_argument('--width', type=int, default=640, help='Default width')
    parser.add_argument('--valid_iters', type=int, default=8)
    parser.add_argument('--max_disp', type=int, default=192)
    
    args = parser.parse_args()
    
    assert args.height % 32 == 0, height must be divisible by 32
    assert args.width % 32 == 0, width must be divisible by 32
    
    os.makedirs(args.save_path, exist_ok=True)
    
    export_dynamic_onnx(args)
