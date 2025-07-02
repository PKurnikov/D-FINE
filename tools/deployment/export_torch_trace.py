import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import torch
import torch.nn as nn
from src.core import YAMLConfig


def main(args):
    # Load config and checkpoint
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

    # Wrap model + postprocessor
    class ExportModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            results = self.postprocessor(outputs, orig_target_sizes)
            return results  # Tuple, determined by return_order

    model = ExportModel().eval().to("cuda:0")
    dummy_input = dummy_input.to("cuda:0")
    dummy_sizes = dummy_sizes.to("cuda:0")

    # Trace the model
    with torch.no_grad():
        traced = torch.jit.trace(model, (dummy_input, dummy_sizes), strict=True)

    output_file = args.resume.replace(".pth", ".pt") if args.resume else "model.pt"
    traced.save(output_file)

    print(f"TorchScript traced model saved to: {output_file}")


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
    args = parser.parse_args()
    main(args)