import copy

import torch
import torchvision
import torchvision.transforms as T

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

# Post training dynamic quantization
model_int8 = torch.ao.quantization.quantize_dynamic(
    model_tmp,
    {
        torch.nn.Linear,
        torch.nn.Conv2d,
        # torch.nn.BatchNorm2d,
        # torch.nn.MaxPool2d,
    },
    dtype=torch.qint8,
    inplace=True,
)


# run the model
# input_fp32 = torch.randn(4, 3, 256, 128)
res_float = model_float(example_inputs)
res_quant = model_int8(example_inputs)
print(res_float)
print(res_quant)

# torch.save(model_int8, "../weights/model_quant.pth.tar")
# torch.save(model_int8.state_dict(), "model_quant.pth.tar")
torch.save(model_float, "../weights/test.pth.tar")
