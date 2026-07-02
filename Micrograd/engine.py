import math

class Value:
    """
    Represents a single scalar value and its gradient, forming the core element
    of our computational graph for automatic differentiation.
    """
    def __init__(self, data, _children=(), _op="", label="", grad=0):
        self.data = data
        self.prev = set(_children)  # Children nodes that produced this value
        self.op = _op               # The operation that produced this node
        self.label = label          # Optional label for visualization
        self.grad = grad            # The derivative of the output (loss) w.r.t. this value
        self._backward = lambda: None  # Callback to propagate gradients to children
        
    def __repr__(self):
        return f"Value: {self.data} | Label: {self.label}"
    
    def __add__(self, other):
        # Allow addition with scalar constants by wrapping them in Value objects
        other = other if isinstance(other, Value) else Value(other) 
        out = Value(self.data + other.data, (self, other), "+")
        
        def _backward():
            # For addition, the gradient flows directly back to both inputs
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        # Handles reverse addition (e.g. constant + Value)
        other = other if isinstance(other, Value) else Value(other) 
        out = Value(self.data + other.data, (self, other), "+")
        
        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
        out._backward = _backward
        return out
    
    def __mul__(self, other): 
        # Allow multiplication with scalar constants
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        
        def _backward():
            # Applying product rule: d(u*v)/du = v, d(u*v)/dv = u
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __truediv__(self, other):
        # Allow division with scalar constants
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), "/")
        
        def _backward():
            # Applying quotient rule:
            # d(u/v)/du = 1/v
            # d(u/v)/dv = -u / (v^2)
            self.grad += out.grad / other.data
            other.grad += out.grad * (-1.0 * (self.data / (other.data ** 2)))
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        # Allow subtraction with scalar constants
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), "-")
        
        def _backward():
            # For subtraction, gradient flows directly to self, and is negated for other
            self.grad += out.grad * 1.0
            other.grad += out.grad * -1.0
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        # Handles raising a Value to a constant power (int or float)
        assert isinstance(other, (int, float)), "power must be int/float"
        out = Value(self.data ** other, (self, ), "**")

        def _backward():
            # Power rule: d(u^n)/du = n * u^(n-1)
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        # Hyperbolic tangent activation function
        x = self.data
        v = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(v, (self, ), "tanh")
        
        def _backward():
            # Derivative of tanh(x) is 1 - tanh^2(x)
            self.grad += (1.0 - v ** 2) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        # Exponential function
        out = Value(math.exp(self.data), (self, ), "exp")
        
        def _backward():
            # Derivative of exp(x) is exp(x)
            self.grad += out.data * out.grad
        out._backward = _backward   
        return out    
    
    def backward(self):
        """
        Runs backward propagation starting from this node.
        Performs a topological sort to compute gradients in the correct dependency order.
        """
        topo = []
        visited = set()
        
        # Helper to construct a topologically ordered list of all ancestor nodes
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build(child)
                topo.append(v)

        build(self)
        
        # Initialize gradient of the root node to 1.0 (dLoss/dLoss = 1.0)
        self.grad = 1.0
        
        # Traverse the computational graph in reverse topological order and trigger callbacks
        for i in reversed(topo):
            i._backward()

