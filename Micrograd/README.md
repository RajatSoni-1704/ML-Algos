# Micrograd from Scratch 🧠

A lightweight, scratch-built educational scalar autograd engine and Multi-Layer Perceptron (MLP) library, inspired by Andrej Karpathy's `micrograd`.

This repository contains a full implementation of backpropagation (reverse-mode automatic differentiation) over a dynamically built DAG (Directed Acyclic Graph) of mathematical operations, along with neural network building blocks and graph visualization utilities.

---

## 📂 Repository Structure

| File                                                  | Description                                                                                                                                                                           |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🛠️[**`engine.py`**](./engine.py)               | Contains the core`Value` class which implements scalar automatic differentiation, topological sorting, and math operations (`+`, `-`, `*`, `/`, `**`, `tanh`, `exp`). |
| 🧠[**`neural_network.py`**](./neural_network.py) | Defines`Neuron`, `Layer`, and `MLP` classes (inheriting from a base `Module`) to build, initialize, and train fully connected neural networks.                                |
| 📊[**`visualize.py`**](./visualize.py)           | Provides utility functions (`track_tree` and `draw_graph`) using `graphviz` to trace the computation graph and render it visually.                                              |
| 📓[**`micrograd.ipynb`**](./micrograd.ipynb)     | A Jupyter notebook demonstrating forward/backward passes, backpropagation trace visualization, and basic optimization loops.                                                          |

---

## 🚀 Key Features

1. **Dynamic Computational Graph (DAG)**: Computations are tracked dynamically as you perform mathematical operations on `Value` objects.
2. **Reverse-Mode Autodiff**: A topological sort tracks dependencies, letting you trigger backpropagation with a single `.backward()` call.
3. **From-Scratch MLP**: Build custom multi-layer perceptrons, compute loss, zero gradients, and update parameters using basic gradient descent.
4. **Rich Graph Visualization**: Render mathematical graphs showing node labels, values (data), operators, and calculated gradients.

---

## 💻 Getting Started

### 1. Basic Autograd Example

Here is how to build a basic computational graph, perform a forward pass, and compute gradients:

```python
from engine import Value

# Define scalar variables with labels
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')

# Define expression: e = a * b + c
d = a * b; d.label = 'd'
e = d + c; e.label = 'e'

# Compute hyperbolic tangent activation
o = e.tanh(); o.label = 'o'

# Run backpropagation
o.backward()

# Access gradients
print(f"Gradient w.r.t a: {a.grad}")  # do/da
print(f"Gradient w.r.t b: {b.grad}")  # do/db
```

### 2. Multi-Layer Perceptron Example

Build a neural network with 3 inputs, two hidden layers of size 4, and 1 output:

```python
from neural_network import MLP

# Initialize MLP: 3 inputs -> hidden layers [4, 4] -> 1 output
model = MLP(3, [4, 4, 1])

# Input data
x = [2.0, 3.0, -1.0]

# Forward pass
y_pred = model(x)
print(f"Prediction: {y_pred}")

# Access parameters
print(f"Number of parameters: {len(model.parameters())}")
```

### 3. Graph Visualization

To visualize the graph in a Jupyter notebook:

```python
from visualize import draw_graph

# Generate and display the graphviz Digraph
dot = draw_graph(o)
dot  # Displays graph inline in Jupyter notebooks
```

---

## 📐 Mathematical Operations Supported

All operations automatically track gradients under the hood:

* **Addition**: $a + b$ & reverse addition $b + a$
* **Subtraction**: $a - b$
* **Multiplication**: $a \cdot b$
* **Division**: $a / b$
* **Power**: $a^n$ (where $n$ is a constant integer or float)
* **Exponential**: $e^a$
* **Hyperbolic Tangent**: $\tanh(a)$
