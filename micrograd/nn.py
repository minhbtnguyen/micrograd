import random
from micrograd.engine import Value


class Module:
    """base class — anything with trainable parameters inherits from this"""

    def zero_grad(self):
        # call BEFORE each backward(); engine accumulates grads with += so leftovers from
        # the previous step would otherwise pollute the next update
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        # subclasses override to expose their leaf Values (weights + biases) for the optimizer
        return []


class Neuron(Module):
    """one neuron: y = act(w·x + b), with optional ReLU nonlinearity"""

    def __init__(self, nin, nonlin=True):
        # uniform[-1, 1] is a crude init — fine for a few layers, breaks for deep nets
        # (that's why real frameworks use Kaiming/Xavier scaled by fan-in)
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin  # final layer typically sets this False so output isn't clamped to >= 0

    def __call__(self, x):
        # sum's `start=self.b` folds the bias into the dot product without a separate add node
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    """a row of Neurons all sharing the same input vector"""

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        # unwrap single-output layers so the final MLP returns a scalar Value, not [Value]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        # flatten each neuron's params into one list for the optimizer
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """stack of Layers — e.g. MLP(3, [4, 4, 1]) is 3 -> 4 -> 4 -> 1"""

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        # last layer is linear (nonlin=False) so regression targets aren't clipped to >= 0
        # and classification logits stay unbounded
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        # forward pass: pipe activations through each layer in sequence
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
