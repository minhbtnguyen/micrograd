class Value:
    """stores a single scalar value and its gradient"""

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = (
            lambda: None
        )  # closure that pulls grad from out -> inputs; overwritten by ops
        self._prev = set(
            _children
        )  # direct inputs that produced this node (edges in the DAG)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        # forward: out = self + other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(out)/d(self) = 1, d(out)/d(other) = 1  -> just pass out.grad through
            # += (not =) because a node may be reused in the graph; gradients must accumulate
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        # forward: out = self * other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(self*other)/d(self) = other,  d(self*other)/d(other) = self  -> swap data into each grad
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        # forward: out = self ** other, where other is a constant exponent (not a Value)
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            # power rule: d(x^n)/dx = n * x^(n-1)
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        # forward: clamps negatives to 0
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            # ReLU's derivative is 1 where out>0, else 0  -> a gradient gate
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        # we need every node to appear AFTER its inputs, so that when we walk it in reverse
        # each node's grad is fully accumulated before we propagate it backwards
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        # seed: d(self)/d(self) = 1 — the loss has gradient 1 with respect to itself
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __rtruediv__(self, other):  # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
