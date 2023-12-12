from onnxruntime.quantization import quantize_dynamic

model_fp32 = "../weights/model.onnx"
model_quant = "../weights/model_quant.onnx"
quantized_model = quantize_dynamic(model_fp32, model_quant)
