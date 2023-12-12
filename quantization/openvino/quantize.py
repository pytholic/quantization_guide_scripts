import copy
import os
import re
import subprocess
from typing import List, Optional

import nncf
import openvino.runtime as ov
import torch
import torchvision
import torchvision.transforms as T
from openvino.tools import mo

import torchreid


def run_benchmark(
    model_path: str, shape: Optional[List[int]] = None, verbose: bool = True
) -> float:
    command = f"benchmark_app -m {model_path} -d CPU -api async -t 15"
    if shape is not None:
        command += f' -shape [{",".join(str(x) for x in shape)}]'
    cmd_output = subprocess.check_output(command, shell=True)  # nosec
    if verbose:
        print(*str(cmd_output).split("\\n")[-9:-1], sep="\n")
    match = re.search(r"Throughput\: (.+?) FPS", str(cmd_output))
    return float(match.group(1))


def get_model_size(
    ir_path: str, m_type: str = "Mb", verbose: bool = True
) -> float:
    xml_size = os.path.getsize(ir_path)
    bin_size = os.path.getsize(os.path.splitext(ir_path)[0] + ".bin")
    for t in ["bytes", "Kb", "Mb"]:
        if m_type == t:
            break
        xml_size /= 1024
        bin_size /= 1024
    model_size = xml_size + bin_size
    if verbose:
        print(f"Model graph (xml):   {xml_size:.3f} Mb")
        print(f"Model weights (bin): {bin_size:.3f} Mb")
        print(f"Model size:          {model_size:.3f} Mb")
    return model_size


# Build and load the model
model_float = torchreid.models.build_model(
    name="osnet_x0_25",
    num_classes=1000,
    loss="softmax",
    pretrained=False,
)
torchreid.utils.load_pretrained_weights(model_float, "../weights/model.pth.tar")
model_tmp = copy.deepcopy(model_float)
model_tmp.eval()

# Prepare claibration dataset
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
dataset = torchvision.datasets.ImageFolder(
    "./test_images/",
    transform=T.Compose(
        [
            T.Resize((256, 128)),
            T.ToTensor(),
            normalize,
        ]
    ),
)
sampler = torch.utils.data.SequentialSampler(dataset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, sampler=sampler)


def transform_fn(data_item):
    images, _ = data_item
    return images


calibration_dataset = nncf.Dataset(dataloader, transform_fn)

# Quantize
model_quant = nncf.quantize(model_tmp, calibration_dataset)

# Export
dummy_input = torch.randn(1, 3, 256, 128)
fp32_onnx_path = "./model_fp32.onnx"
int8_onnx_path = "./model_int8.onnx"

torch.onnx.export(
    model_float.cpu(),
    dummy_input,
    fp32_onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "-1"}},
)
ov_model = mo.convert_model(fp32_onnx_path)

torch.onnx.export(
    model_quant.cpu(),
    dummy_input,
    int8_onnx_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "-1"}},
)
ov_quantized_model = mo.convert_model(int8_onnx_path)

fp32_ir_path = "./model_fp32.xml"
ov.serialize(ov_model, fp32_ir_path)
print(f"[1/7] Save FP32 model: {fp32_ir_path}")
fp32_model_size = get_model_size(fp32_ir_path, verbose=True)

int8_ir_path = "./model_int8.xml"
ov.serialize(ov_quantized_model, int8_ir_path)
print(f"[2/7] Save INT8 model: {int8_ir_path}")
int8_model_size = get_model_size(int8_ir_path, verbose=True)

print("[3/7] Benchmark FP32 model:")
fp32_fps = run_benchmark(fp32_ir_path, shape=[1, 3, 256, 128], verbose=True)
print("[4/7] Benchmark INT8 model:")
int8_fps = run_benchmark(int8_ir_path, shape=[1, 3, 256, 128], verbose=True)
