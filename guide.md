I haven't quantized the OD model yet. Mainly it was for the classification models. You can find my scripts here (for onnx and pytorch quantization):https://github.com/pytholic/quantization_guide_scripts
https://github.com/pytholic/window_detection

I followed these links mainly:https://pytorch.org/tutorials/recipes/quantization.html
https://pytorch.org/docs/stable/quantization.html

If my model is more complex, for example YOLO or something, then it would be hard to quantize manually.In that case, I might also consider trying out TensorRT instead of manual quantization.https://github.com/NVIDIA-AI-IOT/torch2trt
https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/
https://github.com/onnx/onnx-tensorrt
https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/
