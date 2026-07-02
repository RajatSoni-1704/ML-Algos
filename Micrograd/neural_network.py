import random
from engine import Value

class Module:
    """
    Base class for all neural network modules.
    Provides utility methods for parameter management and gradient resetting.
    """
    def parameters(self):
        """Returns a list of all parameters (Values) in this module."""
        return []

    def zero_grad(self):
        """Resets the gradients of all parameters to zero."""
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    """
    Represents a single artificial neuron with weights and a bias.
    Uses hyperbolic tangent (tanh) activation by default.
    """
    def __init__(self, nin):
        # Initialize weights and bias randomly in the range [-1, 1]
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1, 1))
        
    def __call__(self, x):
        # Compute w * x + b
        assert len(x) == len(self.weights), "Input size must match weight count"
        final_sum = sum((wi*xi for wi, xi in zip(self.weights, x)), self.bias)
        # Apply non-linear activation
        final_sum = final_sum.tanh()
        return final_sum
    
    def parameters(self):
        # Parameters of a neuron are its weights and its bias
        return self.weights + [self.bias]
    
    def __repr__(self):
        return f"Neuron(inputs={len(self.weights)}, activation='tanh')"


class Layer(Module):
    """
    Represents a layer of neurons.
    """
    def __init__(self, nin, nout):
        # A layer consists of nout neurons, each with nin inputs
        self.neurons = [Neuron(nin) for _ in range(nout)]
        
    def __call__(self, x):
        # Forward pass through each neuron in the layer
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        # Parameters of a layer are the concatenation of parameters of its neurons
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
    def __repr__(self):
        nin = len(self.neurons[0].weights) if self.neurons else 0
        neuron_repr = str(self.neurons[0]) if self.neurons else "None"
        return f"Layer(inputs={nin}, outputs={len(self.neurons)}) composed of [{neuron_repr}]"


class MLP(Module):
    """
    Multi-Layer Perceptron (MLP) fully-connected neural network.
    """
    def __init__(self, nin: int, nouts: list[int]):
        # sz contains the size of the input, all hidden layers, and the output layer
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        # Forward pass sequentially through all layers
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        # Parameters of the MLP are the parameters of all its layers
        return [p for layer in self.layers for p in layer.parameters()]
    
    def forward_pass(self, learning_rate=0.001):
        """
        Performs a basic gradient descent update step on all network parameters.
        """
        params = self.parameters()
        for p in params:
            p.data += -1.0 * (learning_rate * p.grad)
    
    def __repr__(self):
        return f"MLP(\n  " + ",\n  ".join(str(l) for l in self.layers) + "\n)"