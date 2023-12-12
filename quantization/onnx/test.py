import os
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision.transforms as T
from numpy.linalg import norm
from PIL import Image

# Define model names
model = "/Users/3i-a1-2021-15/Developer/projects/pivo-tracking/similarity-metrics/deep-person-reid/utils/quantization/weights/model.onnx"
model_quant = "/Users/3i-a1-2021-15/Developer/projects/pivo-tracking/similarity-metrics/deep-person-reid/utils/quantization/weights/model_quant.onnx"
models = [model, model_quant]

# Define config
test_dir = "./test_images"
pixel_mean = [0.485, 0.456, 0.406]
pixel_std = [0.229, 0.224, 0.225]
transforms = []
transforms += [T.Resize((256, 128))]
transforms += [T.ToTensor()]
transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
preprocess = T.Compose(transforms)

res = []


# Utility functions
def cosine_similarity(arr1, arr2):
    dist = np.dot(arr1, arr2) / (norm(arr1) * norm(arr2))
    return dist


# Load models
for i, model in enumerate(models):
    onnx_model = onnx.load(f"{model}")
    onnx.checker.check_model(model)
    ort_sess = ort.InferenceSession(f"{model}")

    time_list = []

    for path in os.listdir(test_dir):
        fullpath = os.path.join(test_dir, path)
        image = Image.open(fullpath).convert("RGB")

        img = preprocess(image)
        img = torch.unsqueeze(img, 0)

        start = time.perf_counter()
        output = ort_sess.run(None, {"input": img.numpy()})
        end = time.perf_counter()

        output = np.array(output).squeeze(axis=0).squeeze(axis=0)
        res.append(output)
        # break

        print(f"Inference time for image {path} = {end-start}")
        time_list.append(end - start)

    print(f"Average inference time = {sum(time_list) / len(time_list)}")

# print(res[0])
print(cosine_similarity(res[2], res[7]))  # keep gap of 6
