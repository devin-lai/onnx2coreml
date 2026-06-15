---
name: Bug report
about: A conversion that fails or produces wrong output
labels: bug
---

**What happened**

A clear description of the problem.

**Reproducer**

The ONNX model (or a minimal one) and the command or API call that triggers it:

```python
import onnx2coreml as o2c

o2c.convert("model.onnx", format="mlpackage")
```

**Error output**

The full error. Conversion failures raise a structured `Onnx2CoreMLError` subclass — include the message and the op type.

**Environment**

- OS and version:
- Python version:
- onnx2coreml version (`onnx2coreml schema`):
- coremltools / onnx versions:
