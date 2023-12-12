preprocess:
	python -m onnxruntime.quantization.preprocess --input ../weights/model.onnx --output ../weights/model-infer.onnx --skip_optimization True

quantize:
	python run.py --input_model ../weights/model-infer.onnx --output_model ../weights/model_quant.onnx --calibrate_dataset ../calibration_dataset/