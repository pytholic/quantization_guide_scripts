"""
Original code can be found at:
https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/resnet50_data_reader.py
"""

import os

import numpy
import onnxruntime
import torchvision.transforms as T
from onnxruntime.quantization import CalibrationDataReader
from PIL import Image

pixel_mean = [0.485, 0.456, 0.406]
pixel_std = [0.229, 0.224, 0.225]
transforms = []
transforms += [T.Resize((256, 128))]
transforms += [T.ToTensor()]
transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
preprocess = T.Compose(transforms)


def _preprocess_images(
    images_folder: str,
    size_limit=0,
):
    """
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    """
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + "/" + image_name
        pillow_img = Image.open(image_filepath).convert("RGB")
        input_data = preprocess(pillow_img)
        input_data = input_data.numpy()
        nhwc_data = numpy.expand_dims(input_data, axis=0)
        nchw_data = nhwc_data.transpose(0, 1, 2, 3)  # ONNX Runtime standard
        unconcatenated_batch_data.append(nchw_data)
    batch_data = numpy.concatenate(
        numpy.expand_dims(unconcatenated_batch_data, axis=0), axis=0
    )
    return batch_data


class ResNet50DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, None)
        (_, _, height, width) = session.get_inputs()[0].shape

        # Convert image to input data
        self.nhwc_data_list = _preprocess_images(
            calibration_image_folder, preprocess_fn=preprocess, size_limit=0
        )
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [
                    {self.input_name: nhwc_data}
                    for nhwc_data in self.nhwc_data_list
                ]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None
