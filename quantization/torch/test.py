import os
import time

import numpy as np
import torch
import torchvision.transforms as T
from numpy.linalg import norm
from PIL import Image

import torchreid


def cosine_similarity(arr1, arr2):
    dist = np.dot(arr1, arr2) / (norm(arr1) * norm(arr2))
    return dist


model_orig_path = "../weights/model.pth.tar"
model_quant_path = "../weights/model_quant.pth.tar"
test_dir = "./test_images/person"
image_list = os.listdir(test_dir)

# Define config
pixel_mean = [0.485, 0.456, 0.406]
pixel_std = [0.229, 0.224, 0.225]
transforms = []
transforms += [T.Resize((256, 128))]
transforms += [T.ToTensor()]
transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
preprocess = T.Compose(transforms)

# For original model
model_orig = torchreid.models.build_model(
    name="osnet_x0_25",
    num_classes=1,
    loss="softmax",
    pretrained=False,
)
torchreid.utils.load_pretrained_weights(model_orig, model_orig_path)
model_orig.eval()

# For quantized model
model_quant = torch.load(model_quant_path, map_location="cpu")
model_quant.eval()

models = [model_orig, model_quant]

res = []

for model in models:
    num_params, flops = torchreid.utils.compute_model_complexity(
        model, (1, 3, 256, 128)
    )
    print(f"Total params: {num_params / 10**6}")  # million
    print(f"Total flops: {flops / 10**9}")  # gigaflops

    time_list = []

    for path in image_list:
        fullpath = os.path.join(test_dir, path)
        print(fullpath)

        image = Image.open(fullpath).convert("RGB")
        img = preprocess(image)
        img = img.unsqueeze(0)

        start = time.perf_counter()
        with torch.no_grad():
            features = model(img)
        end = time.perf_counter()

        res.append(features.squeeze())
        # print(features)
        # break

        print(f"Inference time for image {path} = {end-start}")
        time_list.append(end - start)

    print(f"Average inference time = {sum(time_list) / len(time_list)}")

print(cosine_similarity(res[2], res[8]))  # keep gap of 6
