# Methods to save and load quantized model
1. Using `state_dict`

**Saving**
```python3
backend = "fbgemm"  # fbgemm, qnnpack
model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend
model_static_quantized = torch.quantization.prepare(model, inplace=False)
input_ = torch.randn(1, 3, 224, 224)
model_static_quantized(input_)
model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)

torch.save(mdl.state_dict(), os.path.join(os.getcwd(), "model", "quant_model.pth"))
```

**Loading**
```python3
model = Net()
backend = "fbgemm"
model.qconfig = torch.quantization.get_default_qconfig(backend)
model_prepared = torch.quantization.prepare(model, inplace=False)
input_ = torch.randn(1, 3, 224, 224)
model_prepared(input_)
quant_model = torch.quantization.convert(model_prepared) 
    
state_dict = torch.load(os.getcwd() + '/model/quant_model_qnn.pth')
quant_model.load_state_dict(state_dict, strict=False)
quant_model.eval()
```

2. Use `jit.save` and `jit.load`

```python3
torch.jit.save(torch.jit.script(model_static_quantized), "./model/quantized_test.pt")
quantized = torch.jit.load("./model/quantized_test.pt")
```
