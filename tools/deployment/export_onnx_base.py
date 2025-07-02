"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import torch
import torch.nn as nn
from src.core import YAMLConfig


def main(args):
    # Load config and model
    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        state_dict = checkpoint.get("ema", {}).get("module") or checkpoint["model"]
        cfg.model.load_state_dict(state_dict)
    else:
        print("Warning: model state_dict not loaded. Using default initialization.")

    # Dummy image input W x H
    scale_det = 4
    dummy_img = torch.randn(1, 3, 576, 1024)
    dummy_sizes = torch.tensor([[1024 // scale_det, 576 // scale_det]], dtype=torch.float)

    # Build wrapped model
    class ExportModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            self.register_buffer(f'orig_target_sizes', torch.tensor([1024 // 4, 576 // 4]))

        def forward(self, images):
            outputs = self.model(images)
            results = self.postprocessor(outputs, self.orig_target_sizes)
            return results  # returns a tuple, based on postprocessor.return_order

    model = ExportModel()
    model.eval()

    # Prepare export params
    output_names = cfg.postprocessor.return_order
    dynamic_axes = {
        "images": {0: "batch"},
        "orig_target_sizes": {0: "batch"}
    }
    for name in output_names:
        dynamic_axes[name] = {0: "batch"}

    # Export to ONNX
    output_file = args.resume.replace(".pth", ".onnx") if args.resume else "model.onnx"
    torch.onnx.export(
        model,
        dummy_img,
        output_file,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=None,
        opset_version=16,
        do_constant_folding=True,
        verbose=False,
    )

    # Optional: Check model
    if args.check:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print("ONNX model check passed.")

    # Optional: Simplify model
    if args.simplify:
        import onnx
        import onnxsim
        input_shapes = {
            "images": dummy_img.shape
        }
        model_simplified, success = onnxsim.simplify(output_file, test_input_shapes=input_shapes)
        if success:
            onnx.save(model_simplified, output_file)
            print("ONNX model simplified successfully.")
        else:
            print("Warning: Simplification failed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/dfine/objects365/dfine_hgnetv2_m_obj2coco.py"
    )
    parser.add_argument(
        "--resume", "-r",
        type=str,
        help="Path to checkpoint (.pth)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        default=True,
        help="Check exported ONNX model"
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=True,
        help="Simplify ONNX model with onnxsim"
    )
    args = parser.parse_args()
    main(args)
