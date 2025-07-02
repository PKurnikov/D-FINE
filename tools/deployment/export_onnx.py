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


def main(
    args,
):
    """main"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    input_shape = cfg.yaml_cfg.get("input_shape", [576, 1024])
    assert isinstance(input_shape, (list, tuple)) and len(input_shape) == 2, "input_shape must be [H, W]"


    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print("not load model.state_dict, use default init state dict...")

    W = 1024 // 4
    H = 576 // 4

    class Model(nn.Module):
        def __init__(
            self,
        ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            self.register_buffer(f'orig_target_sizes', torch.tensor([W, H]))

        def forward(self, images, orig_target_sizes=None):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes = self.orig_target_sizes)
            return outputs

    model = Model()

    # Prepare dummy input
    H, W = input_shape
    data = torch.rand(1, 3, H, W)
    size = torch.tensor([[W // 4, H // 4]])

    # _ = model(data, size)

    # Torch trace
    do_torch = False
    if (do_torch):
        model.eval()
        output_file = args.resume.replace(".pth", ".pt") if args.resume else "model.pt"
        #Trace model for torch server deploy
        with torch.no_grad():
            # _ = model.forward(data.to('cuda:0'))
            # model_traced = torch.jit.trace(model, torch.randn(1, 3, H, W, requires_grad=False).to('cuda:0'))
            model_traced = torch.jit.trace(model.to('cuda:0'), data.to('cuda:0'), strict=True)
        model_traced.save(output_file)

    # data = torch.rand(1, 3, 640, 640)
    # size = torch.tensor([[640, 640]])
    # _ = model(data, size)

    dynamic_axes = {
        "images": {
            0: "N",
        },
        "orig_target_sizes": {0: "N"},
    }

    output_file = args.resume.replace(".pth", "_op16.onnx") if args.resume else "model.onnx"

    torch.onnx.export(
        model,
        data,
        output_file,
        input_names=["images"],
        output_names=["segmentation", "detections"],
        dynamic_axes = None,
        # dynamic_axes={
        #     "images": {0: "N"},
        #     "orig_target_sizes": {0: "N"}
        # },
        opset_version=16,
        verbose=False,
        do_constant_folding=True,
    )

    if args.check:
        import onnx

        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print("Check export onnx model done...")

    if args.simplify:
        import onnx
        import onnxsim

        dynamic = False
        # input_shapes = {'images': [1, 3, 640, 640], 'orig_target_sizes': [1, 2]} if dynamic else None
        input_shapes = {"images": data.shape} if dynamic else None
        onnx_model_simplify, check = onnxsim.simplify(output_file, test_input_shapes=input_shapes)
        onnx.save(onnx_model_simplify, output_file)
        print(f"Simplify onnx model {check}...")

def find_files(catalog, extensions):
    """
    Самый быстрый вариант поиска файлов с расширениями в каталоге и подкаталогах.
    """
    return [str(p) for ext in extensions for p in Path(catalog).rglob(f'*{ext}')]

if __name__ == "__main__":


    # import shutil

    # with open('/data1/Datasets/shared_datasets/layer_3_lists/val.txt', 'r') as fp:
    #     for line in fp:
    #         print(line.rstrip('\n'))
    #         src_path, gt_path = line.split('\t')
    #         src_path = os.path.join('/data1/Datasets/shared_datasets/layer_3_lists', src_path)
    #         gt_path = os.path.join('/data1/Datasets/shared_datasets/layer_3_lists', gt_path[:-1])

    #         shutil.copy(src_path, os.path.join('/data/Datasets/tractor_shared/segm/agro_drivable/leftImg8bit/val/Agro', os.path.basename(src_path)))
    #         shutil.copy(gt_path, os.path.join('/data/Datasets/tractor_shared/segm/agro_drivable/gtFine/val/Agro', os.path.basename(gt_path)))

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="configs/dfine/dfine_hgnetv2_l_coco.yml",
        type=str,
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=True,
    )
    args = parser.parse_args()
    main(args)