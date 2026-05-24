<div align="center">

# Micrograd

**A tiny scalar autograd engine and neural-net library, built from scratch.**

</div>

---

## What's inside

- **`micrograd/engine.py`** - the `Value` class: scalar autograd via Python operator overloads.
- **`micrograd/nn.py`** - neural-net building blocks: `Module` to `Neuron` to `Layer` to `MLP`.
- **`notebooks/`** - step-by-step learning notebooks (gradients, MLP from scratch, PyTorch comparison).
- **`demo.ipynb`** - train an MLP on `make_moons`.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name micrograd --display-name "Python (micrograd)"
jupyter lab
```

## Quick taste

```python
from micrograd.engine import Value
from micrograd.nn import MLP

model = MLP(2, [16, 16, 1])     # 2 -> 16 -> 16 -> 1
y = model([Value(2.0), Value(-1.0)])
y.backward()                    # gradients flow to every weight and bias
```

## Reference

- Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).
