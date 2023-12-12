import copy

import torch
import torchvision
import torchvision.transforms as T
from torch.ao.quantization import QConfigMapping, get_default_qconfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

import torchreid

# Build and load the model
model_float = torchreid.models.build_model(
    name="osnet_x0_25",
    num_classes=1000,
    loss="softmax",
    pretrained=False,
)
torchreid.utils.load_pretrained_weights(model_float, "../weights/model.pth.tar")
model_tmp = copy.deepcopy(model_float)
# model_tmp.eval()

# Prepare calibration data
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
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=8, sampler=sampler
)
example_inputs = next(iter(data_loader))[0]  # get an example input
print(example_inputs.shape)

# Post training static quantization

qconfig = get_default_qconfig("x86")
qconfig_mapping = QConfigMapping().set_global(qconfig)


def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            model(image)


prepared_model = prepare_fx(
    model_float, qconfig_mapping, example_inputs
)  # fuse modules and insert observers
model_quant = convert_fx(prepared_model)
print(model_quant)
