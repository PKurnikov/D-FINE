import tensorrt as trt
import os
import sys

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def get_trt_major_version():
    return int(trt.__version__.split('.')[0])

version = get_trt_major_version()

def convert_onnx_to_trt(onnx_path, engine_path, batch_size=1, precision="fp32"):
    trt_version = get_trt_major_version()
    print(f"[INFO] Detected TensorRT version: {trt.__version__}")

    # Initialize TensorRT stuff
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # Parse ONNX model
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, 'rb') as model:
        parser.parse(model.read())
        
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    # Create optimization profile
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "images",
        min=(1, 3, 576, 1024),
        opt=(batch_size, 3, 576, 1024),
        max=(batch_size, 3, 576, 1024)
    )
    profile.set_shape(
        "orig_target_sizes",
        min=(1, 2),
        opt=(batch_size, 2),
        max=(batch_size, 2)
    )

    # Configure builder
    config = builder.create_builder_config()

    # if trt_version >= 10:
    #     config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
    # else:
    #     config.max_workspace_size = 2 << 30  # 2GB

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB

    # for tensorrt 8
    # config.max_workspace_size = 2 << 30  # 2GB

    # for tensorrt 10
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30) # 2GB

    config.add_optimization_profile(profile)

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        # config.clear_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        # config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        
        for layer_idx in range(network.num_layers):
            layer = network[layer_idx]
            if layer.type == trt.LayerType.CONVOLUTION:
                print(layer.type)
                layer.precision = trt.float32
                layer.set_output_type(0, trt.float32)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    # config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    # config.set_flag(trt.BuilderFlag.DIRECT_IO)
    # config.set_flag(trt.BuilderFlag.REJECT_EMPTY_ALGORITHMS)
    # config.clear_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    # config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    # if trt_version < 10:
    # config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS) #STRICT_TYPES
    #     engine = builder.build_engine(network, config)
    #     if engine is None:
    #         print("[ERROR] Failed to build engine.")
    #         return
    #     with open(engine_path, 'wb') as f:
    #         f.write(engine.serialize())
    # else:
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        print("[ERROR] Failed to build serialized engine.")
        return
    with open(engine_path, 'wb') as f:
        f.write(engine)

    # for tensorrt 8
    # config.set_flag(trt.BuilderFlag.STRICT_TYPES)

    # Build and save engine
    # for tensorrt 8
    # engine = builder.build_engine(network, config)

    # for tensorrt 10
    # engine = builder.build_serialized_network(network, config)

    # for tensorrt 10
    # with open(engine_path, 'wb') as f:
    #     f.write(engine)

    # for tensorrt 8
    # with open(engine_path, 'wb') as f:
    #     f.write(engine.serialize())

    print(f"Successfully converted model to TensorRT engine: {engine_path}")


# if __name__ == "__main__":
#     # Example usage
#     convert_onnx_to_trt(
#         onnx_path="output/dfine_hgnetv2_s_obj2coco/checkpoint0010.onnx",
#         engine_path="output/dfine_hgnetv2_s_obj2coco/checkpoint0010.engine",
#         batch_size=1,
#         precision="fp16"
#     )
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert ONNX model to TensorRT engine.")
    parser.add_argument("--onnx", type=str, required=True, help="Path to the ONNX model file.")
    parser.add_argument("--trt", type=str, required=True, help="Path where the TensorRT engine will be saved.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the engine.")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32", help="Precision mode (default: fp16)")

    args = parser.parse_args()

    convert_onnx_to_trt(
        onnx_path=args.onnx,
        engine_path=args.trt,
        batch_size=args.batch_size,
        precision=args.precision
    )