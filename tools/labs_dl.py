"""Guided browser labs (NumPy) for Deep Learning weekly tasks DL1-DL12 and project P07."""
from labs_warmup import S

def F(**kw):
    """A read-and-run step with no blanks."""
    d = S(**kw); d['free'] = True; return d

DL = [
dict(id='DL1', title='The perceptron and its convergence bound', packages=['numpy', 'matplotlib'], intro=
"The perceptron is the oldest learning rule: when you get a point wrong, nudge the weights toward it. On separable data it always stops, and the proof even bounds how many mistakes it can make.",
steps=[
S('The learning rule',
"Labels are +1/-1. A point is misclassified when y (w . x) <= 0. Then update w = w + y x. Loop over the data until a full pass makes no mistakes.",
'''
import numpy as np
rng = np.random.default_rng(0)
w_true = np.array([1.0, -2.0])
X = rng.uniform(-1, 1, (200, 2))
margin = X @ w_true / np.linalg.norm(w_true)
X, margin = X[np.abs(margin) > 0.1], margin[np.abs(margin) > 0.1]   # keep a clear gap
y = np.sign(margin)

def perceptron(X, y, max_epochs=100):
    w, mistakes = np.zeros(X.shape[1]), 0
    for _ in range(max_epochs):
        clean = True
        for xi, yi in zip(X, y):
            if yi * (w @ xi) <= 0:
                w = ___
                mistakes += 1
                clean = False
        if clean:
            break
    return w, mistakes

w, mistakes = perceptron(X, y)
print(w, mistakes, np.mean(np.sign(X @ w) == y))
''','''
import numpy as np
rng = np.random.default_rng(0)
w_true = np.array([1.0, -2.0])
X = rng.uniform(-1, 1, (200, 2))
margin = X @ w_true / np.linalg.norm(w_true)
X, margin = X[np.abs(margin) > 0.1], margin[np.abs(margin) > 0.1]   # keep a clear gap
y = np.sign(margin)

def perceptron(X, y, max_epochs=100):
    w, mistakes = np.zeros(X.shape[1]), 0
    for _ in range(max_epochs):
        clean = True
        for xi, yi in zip(X, y):
            if yi * (w @ xi) <= 0:
                w = w + yi * xi
                mistakes += 1
                clean = False
        if clean:
            break
    return w, mistakes

w, mistakes = perceptron(X, y)
print(w, mistakes, np.mean(np.sign(X @ w) == y))
''','''
assert np.mean(np.sign(X @ w) == y) == 1.0, "The perceptron should separate the data perfectly"
''', "w + yi * xi"),
S('Check the theorem',
"Novikoff's theorem: mistakes <= (R / gamma)^2, where R is the largest ||x|| and gamma is the smallest margin y (w* . x) / ||w*|| for a separating w*. Compute the bound with w_true.",
'''
R = np.max(np.linalg.norm(X, axis=1))
gamma = np.min(y * (X @ w_true)) / np.linalg.norm(w_true)
bound = ___
print(mistakes, bound)
''','''
R = np.max(np.linalg.norm(X, axis=1))
gamma = np.min(y * (X @ w_true)) / np.linalg.norm(w_true)
bound = (R / gamma) ** 2
print(mistakes, bound)
''','''
assert abs(bound - (R / gamma) ** 2) < 1e-9 and mistakes <= bound, "mistakes should be at most (R / gamma)^2"
''', "(R / gamma) ** 2"),
]),

dict(id='DL2', title="What one perceptron can't do: XOR", packages=['numpy'], intro=
"A single perceptron draws one straight line. Of the 16 Boolean functions of two inputs, exactly two can't be drawn that way. Find them, then fix XOR with a second layer.",
steps=[
S('Which functions are linearly separable?',
"Enumerate all 16 truth tables. For each, run the perceptron (with a bias input of 1) for 100 epochs; if it never gets a clean pass, the function isn't separable.",
'''
import numpy as np
from itertools import product
inputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])   # last column is the bias

def separable(targets):
    y = np.where(np.array(targets) == 1, 1, -1)
    w = np.zeros(3)
    for _ in range(100):
        errors = 0
        for x, t in zip(inputs, y):
            if t * (w @ x) <= 0:
                w += t * x
                errors += 1
        if errors == 0:
            return True
    return False

not_separable = [t for t in product([0, 1], repeat=4) if ___]
print(not_separable)
''','''
import numpy as np
from itertools import product
inputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])   # last column is the bias

def separable(targets):
    y = np.where(np.array(targets) == 1, 1, -1)
    w = np.zeros(3)
    for _ in range(100):
        errors = 0
        for x, t in zip(inputs, y):
            if t * (w @ x) <= 0:
                w += t * x
                errors += 1
        if errors == 0:
            return True
    return False

not_separable = [t for t in product([0, 1], repeat=4) if not separable(t)]
print(not_separable)
''','''
assert sorted(not_separable) == [(0, 1, 1, 0), (1, 0, 0, 1)], "Exactly XOR (0,1,1,0) and XNOR (1,0,0,1) should fail"
''', "Keep the tables where separable(t) is False"),
S('XOR from two layers',
"XOR(a, b) = AND(OR(a, b), NAND(a, b)). Each gate is one perceptron: step(w . x + b). Fill in the output layer's weights and bias so it computes AND of the two hidden units.",
'''
step = lambda z: (z > 0).astype(int)
def xor_net(a, b):
    x = np.array([a, b])
    h_or = step(np.array([1, 1]) @ x - 0.5)
    h_nand = step(np.array([-1, -1]) @ x + 1.5)
    h = np.array([h_or, h_nand])
    return int(step(np.array(___) @ h + ___))

print([xor_net(a, b) for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]])
''','''
step = lambda z: (z > 0).astype(int)
def xor_net(a, b):
    x = np.array([a, b])
    h_or = step(np.array([1, 1]) @ x - 0.5)
    h_nand = step(np.array([-1, -1]) @ x + 1.5)
    h = np.array([h_or, h_nand])
    return int(step(np.array([1, 1]) @ h + -1.5))

print([xor_net(a, b) for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]])
''','''
assert [xor_net(a, b) for a, b in [(0, 0), (0, 1), (1, 0), (1, 1)]] == [0, 1, 1, 0], "The network should output 0, 1, 1, 0"
''', "AND fires only when both are 1: weights [1, 1], bias -1.5"),
]),

dict(id='DL3', title='Build autograd: micrograd in 40 lines', packages=[], intro=
"PyTorch's magic is automatic differentiation: every number remembers how it was made, so gradients can flow backwards. You'll build a tiny version (after Andrej Karpathy's micrograd) and train a neuron with it.",
steps=[
S('A Value that remembers its parents',
"Each operation creates a new Value and stores a small `_backward` function that adds its share of gradient to its inputs. For a * b: a.grad += b.data * out.grad, b.grad += a.data * out.grad.",
'''
import math

class Value:
    def __init__(self, data, parents=()):
        self.data, self.grad = data, 0.0
        self._parents, self._backward = parents, lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += ___
            other.grad += ___
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,))
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    __radd__ = __add__
    __rmul__ = __mul__
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
''','''
import math

class Value:
    def __init__(self, data, parents=()):
        self.data, self.grad = data, 0.0
        self._parents, self._backward = parents, lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,))
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    __radd__ = __add__
    __rmul__ = __mul__
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
''','''
a, b = Value(3.0), Value(-2.0)
c = a * b
c.grad = 1.0
c._backward()
assert (a.grad, b.grad) == (-2.0, 3.0), f"For c = a*b, dc/da = b = -2 and dc/db = a = 3; you have {a.grad}, {b.grad}"
''', "Each input's gradient is the OTHER input's value times out.grad"),
S('backward(): visit the graph in reverse',
"To run every `_backward` in the right order, sort the graph so each node comes after its parents (a topological sort), then walk it in reverse.",
'''
def backward(root):
    order, seen = [], set()
    def visit(v):
        if v not in seen:
            seen.add(v)
            for p in v._parents:
                visit(p)
            order.append(v)
    visit(root)
    root.grad = 1.0
    for v in ___:
        v._backward()

x, w, b = Value(2.0), Value(-3.0), Value(1.0)
y = (x * w + b).tanh()
backward(y)
print(y.data, x.grad, w.grad, b.grad)
''','''
def backward(root):
    order, seen = [], set()
    def visit(v):
        if v not in seen:
            seen.add(v)
            for p in v._parents:
                visit(p)
            order.append(v)
    visit(root)
    root.grad = 1.0
    for v in reversed(order):
        v._backward()

x, w, b = Value(2.0), Value(-3.0), Value(1.0)
y = (x * w + b).tanh()
backward(y)
print(y.data, x.grad, w.grad, b.grad)
''','''
d = 1 - math.tanh(-5.0) ** 2
assert abs(w.grad - 2.0 * d) < 1e-12 and abs(x.grad - (-3.0) * d) < 1e-12, "Walk the topological order in reverse: reversed(order)"
''', "reversed(order)"),
S('Train a neuron with your engine',
"Learn y = tanh(w x + b) to fit four points. Each step: build the loss, zero the grads, backward, nudge w and b downhill.",
'''
data = [(-2.0, -0.9), (-1.0, -0.6), (1.0, 0.6), (2.0, 0.9)]
w, b = Value(0.1), Value(0.0)
for step in range(200):
    loss = Value(0.0)
    for xi, yi in data:
        err = (w * xi + b).tanh() - yi
        loss = loss + err * err
    w.grad = b.grad = 0.0
    backward(loss)
    w.data -= 0.1 * ___
    b.data -= 0.1 * b.grad
print(loss.data, w.data, b.data)
''','''
data = [(-2.0, -0.9), (-1.0, -0.6), (1.0, 0.6), (2.0, 0.9)]
w, b = Value(0.1), Value(0.0)
for step in range(200):
    loss = Value(0.0)
    for xi, yi in data:
        err = (w * xi + b).tanh() - yi
        loss = loss + err * err
    w.grad = b.grad = 0.0
    backward(loss)
    w.data -= 0.1 * w.grad
    b.data -= 0.1 * b.grad
print(loss.data, w.data, b.data)
''','''
assert loss.data < 0.01, f"Loss should get below 0.01, you have {loss.data:.4f}"
''', "w.data -= 0.1 * w.grad"),
]),

dict(id='DL4', title='Optimiser race on the Rosenbrock valley', packages=['numpy', 'matplotlib'], intro=
"Rosenbrock's banana-shaped valley is the classic test for optimisers. Implement SGD, momentum, RMSProp and Adam and race them.",
steps=[
S('The function and its gradient',
"f(x, y) = (1 - x)^2 + 100 (y - x^2)^2, minimum at (1, 1). Fill in the y-component of the gradient.",
'''
import numpy as np
import matplotlib.pyplot as plt
def f(p):
    x, y = p
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def grad(p):
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x ** 2), ___])
print(grad(np.array([1.0, 1.0])))
''','''
import numpy as np
import matplotlib.pyplot as plt
def f(p):
    x, y = p
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def grad(p):
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x ** 2), 200 * (y - x ** 2)])
print(grad(np.array([1.0, 1.0])))
''','''
p0 = np.array([-0.7, 1.3]); h = 1e-6
num = np.array([(f(p0 + [h, 0]) - f(p0 - [h, 0])) / (2 * h), (f(p0 + [0, h]) - f(p0 - [0, h])) / (2 * h)])
assert np.allclose(grad(p0), num, rtol=1e-4), "The y-gradient is d/dy of 100 (y - x^2)^2"
''', "200 * (y - x ** 2)"),
S('Four optimisers',
"Adam keeps running averages of the gradient (m) and its square (v), corrects their start-up bias, and steps by m_hat / (sqrt(v_hat) + eps).",
'''
def run(update, steps=3000, start=(-1.2, 1.0)):
    p = np.array(start); state = {}
    path = [p.copy()]
    for t in range(1, steps + 1):
        p = update(p, grad(p), state, t)
        path.append(p.copy())
    return np.array(path)

def sgd(p, g, s, t, lr=1e-3):
    return p - lr * g

def momentum(p, g, s, t, lr=1e-3, beta=0.9):
    s["v"] = beta * s.get("v", 0) - lr * g
    return p + s["v"]

def rmsprop(p, g, s, t, lr=1e-3, beta=0.9, eps=1e-8):
    s["v"] = beta * s.get("v", 0) + (1 - beta) * g ** 2
    return p - lr * g / (np.sqrt(s["v"]) + eps)

def adam(p, g, s, t, lr=2e-2, b1=0.9, b2=0.999, eps=1e-8):
    s["m"] = b1 * s.get("m", 0) + (1 - b1) * g
    s["v"] = b2 * s.get("v", 0) + (1 - b2) * g ** 2
    m_hat = s["m"] / (1 - b1 ** t)
    v_hat = ___
    return p - lr * m_hat / (np.sqrt(v_hat) + eps)

paths = {name: run(fn) for name, fn in [("SGD", sgd), ("momentum", momentum), ("RMSProp", rmsprop), ("Adam", adam)]}
for name, path in paths.items():
    print(f"{name:9s} final {path[-1].round(3)}  f = {f(path[-1]):.5f}")
''','''
def run(update, steps=3000, start=(-1.2, 1.0)):
    p = np.array(start); state = {}
    path = [p.copy()]
    for t in range(1, steps + 1):
        p = update(p, grad(p), state, t)
        path.append(p.copy())
    return np.array(path)

def sgd(p, g, s, t, lr=1e-3):
    return p - lr * g

def momentum(p, g, s, t, lr=1e-3, beta=0.9):
    s["v"] = beta * s.get("v", 0) - lr * g
    return p + s["v"]

def rmsprop(p, g, s, t, lr=1e-3, beta=0.9, eps=1e-8):
    s["v"] = beta * s.get("v", 0) + (1 - beta) * g ** 2
    return p - lr * g / (np.sqrt(s["v"]) + eps)

def adam(p, g, s, t, lr=2e-2, b1=0.9, b2=0.999, eps=1e-8):
    s["m"] = b1 * s.get("m", 0) + (1 - b1) * g
    s["v"] = b2 * s.get("v", 0) + (1 - b2) * g ** 2
    m_hat = s["m"] / (1 - b1 ** t)
    v_hat = s["v"] / (1 - b2 ** t)
    return p - lr * m_hat / (np.sqrt(v_hat) + eps)

paths = {name: run(fn) for name, fn in [("SGD", sgd), ("momentum", momentum), ("RMSProp", rmsprop), ("Adam", adam)]}
for name, path in paths.items():
    print(f"{name:9s} final {path[-1].round(3)}  f = {f(path[-1]):.5f}")
''','''
assert f(paths["Adam"][-1]) < f(paths["SGD"][-1]), "Adam should end lower than plain SGD"
assert f(paths["momentum"][-1]) < f(paths["SGD"][-1]), "Momentum should beat plain SGD"
''', "Same correction as m_hat, with b2: s['v'] / (1 - b2 ** t)"),
F(title='Plot the race',
text="Draw the paths over the contour map. Notice SGD crawling along the valley floor while the others sweep through it.",
starter='''
xs, ys = np.meshgrid(np.linspace(-1.5, 1.5, 200), np.linspace(-0.5, 2, 200))
Z = (1 - xs) ** 2 + 100 * (ys - xs ** 2) ** 2
plt.contour(xs, ys, np.log(Z + 1e-3), 30, alpha=0.4)
for name, path in paths.items():
    plt.plot(path[:, 0], path[:, 1], label=name, lw=1.2)
plt.plot(1, 1, "k*", ms=12); plt.legend()
''', solution='''
xs, ys = np.meshgrid(np.linspace(-1.5, 1.5, 200), np.linspace(-0.5, 2, 200))
Z = (1 - xs) ** 2 + 100 * (ys - xs ** 2) ** 2
plt.contour(xs, ys, np.log(Z + 1e-3), 30, alpha=0.4)
for name, path in paths.items():
    plt.plot(path[:, 0], path[:, 1], label=name, lw=1.2)
plt.plot(1, 1, "k*", ms=12); plt.legend()
''', check='''
assert len(plt.gca().lines) >= 5
''', hint="Nothing to fill in here: run it and look."),
]),

dict(id='DL5', title='PCA on handwritten digits, two ways', packages=['numpy', 'scikit-learn', 'matplotlib'], intro=
"PCA via the covariance matrix's eigenvectors and via the SVD of the data give the same components. Confirm it, then look at what the components are.",
steps=[
S('Eigendecomposition vs SVD',
"Centre the 8x8 digit images. Route 1: eigenvectors of the covariance matrix. Route 2: right singular vectors of the data. The top components should match up to sign.",
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
X = load_digits().data
Xc = X - X.mean(axis=0)
cov = Xc.T @ Xc / (len(X) - 1)
evals, evecs = np.linalg.eigh(cov)
order = np.argsort(evals)[::-1]
eig_pcs = evecs[:, order].T                 # rows are components
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
svd_pcs = ___
match = all(abs(abs(eig_pcs[i] @ svd_pcs[i]) - 1) < 1e-6 for i in range(10))
print(match)
''','''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
X = load_digits().data
Xc = X - X.mean(axis=0)
cov = Xc.T @ Xc / (len(X) - 1)
evals, evecs = np.linalg.eigh(cov)
order = np.argsort(evals)[::-1]
eig_pcs = evecs[:, order].T                 # rows are components
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
svd_pcs = Vt
match = all(abs(abs(eig_pcs[i] @ svd_pcs[i]) - 1) < 1e-6 for i in range(10))
print(match)
''','''
assert match, "The rows of Vt are the principal components"
assert np.allclose(s[:5] ** 2 / (len(X) - 1), evals[order][:5]), "Squared singular values / (n-1) are the eigenvalues"
''', "svd_pcs = Vt"),
S('See the components',
"Reshape the top 16 components to 8x8 images. Early ones look like blurry strokes; later ones get noisy.",
'''
fig, axes = plt.subplots(2, 8, figsize=(10, 3))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(svd_pcs[i].reshape(___), cmap="RdBu")
    ax.set_title(f"PC{i+1}", fontsize=8); ax.axis("off")
''','''
fig, axes = plt.subplots(2, 8, figsize=(10, 3))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(svd_pcs[i].reshape(8, 8), cmap="RdBu")
    ax.set_title(f"PC{i+1}", fontsize=8); ax.axis("off")
''','''
assert len(plt.gcf().axes) == 16
''', "Each component has 64 numbers: reshape(8, 8)"),
]),

dict(id='DL6', title='Autoencoders and denoising', packages=['numpy', 'scikit-learn', 'matplotlib'], intro=
"An autoencoder squeezes inputs through a narrow bottleneck and rebuilds them. A denoising autoencoder is trained to rebuild clean inputs from noisy ones. You'll write both in NumPy on the digits data.",
steps=[
S('Forward and backward for a 64-16-64 autoencoder',
"Encoder: h = relu(X W1 + b1). Decoder: X_hat = h W2 + b2. Loss: mean squared error. The backward pass is the same chain rule as IML5.",
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
X = load_digits().data / 16.0
rng = np.random.default_rng(0)

def init():
    return {"W1": rng.normal(0, 0.1, (64, 16)), "b1": np.zeros(16), "W2": rng.normal(0, 0.1, (16, 64)), "b2": np.zeros(64)}

def step(P, X_in, X_target, lr):
    h = np.maximum(0, X_in @ P["W1"] + P["b1"])
    X_hat = h @ P["W2"] + P["b2"]
    n = len(X_in)
    d_out = 2 * (X_hat - X_target) / (n * 64)
    d_h = (d_out @ P["W2"].T) * (h > 0)
    P["W2"] -= lr * h.T @ d_out;   P["b2"] -= lr * d_out.sum(axis=0)
    P["W1"] -= lr * ___;           P["b1"] -= lr * d_h.sum(axis=0)
    return np.mean((X_hat - X_target) ** 2)

P = init()
losses = [step(P, X, X, lr=2.0) for _ in range(800)]
print(losses[0], losses[-1])
plt.plot(losses); plt.xlabel("step"); plt.ylabel("reconstruction MSE")
''','''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
X = load_digits().data / 16.0
rng = np.random.default_rng(0)

def init():
    return {"W1": rng.normal(0, 0.1, (64, 16)), "b1": np.zeros(16), "W2": rng.normal(0, 0.1, (16, 64)), "b2": np.zeros(64)}

def step(P, X_in, X_target, lr):
    h = np.maximum(0, X_in @ P["W1"] + P["b1"])
    X_hat = h @ P["W2"] + P["b2"]
    n = len(X_in)
    d_out = 2 * (X_hat - X_target) / (n * 64)
    d_h = (d_out @ P["W2"].T) * (h > 0)
    P["W2"] -= lr * h.T @ d_out;   P["b2"] -= lr * d_out.sum(axis=0)
    P["W1"] -= lr * X_in.T @ d_h;  P["b1"] -= lr * d_h.sum(axis=0)
    return np.mean((X_hat - X_target) ** 2)

P = init()
losses = [step(P, X, X, lr=2.0) for _ in range(800)]
print(losses[0], losses[-1])
plt.plot(losses); plt.xlabel("step"); plt.ylabel("reconstruction MSE")
''','''
assert losses[-1] < 0.5 * losses[0], "Reconstruction error should at least halve"
''', "The input of this layer is X_in: X_in.T @ d_h"),
S('Denoising',
"Train a second autoencoder whose INPUT is noisy (Gaussian noise, sd 0.3) but whose TARGET is the clean image. Then compare both on fresh noisy inputs.",
'''
P_dn = init()
for _ in range(800):
    noisy = X + rng.normal(0, 0.3, X.shape)
    step(P_dn, ___, ___, lr=2.0)

def reconstruct(P, X_in):
    return np.maximum(0, X_in @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]

test_noisy = X + np.random.default_rng(1).normal(0, 0.3, X.shape)
err_plain = np.mean((reconstruct(P, test_noisy) - X) ** 2)
err_dn = np.mean((reconstruct(P_dn, test_noisy) - X) ** 2)
print(err_plain, err_dn)
fig, axes = plt.subplots(3, 6, figsize=(8, 4))
for j in range(6):
    for i, img in enumerate([test_noisy[j], reconstruct(P, test_noisy[j:j+1])[0], reconstruct(P_dn, test_noisy[j:j+1])[0]]):
        axes[i, j].imshow(img.reshape(8, 8), cmap="gray"); axes[i, j].axis("off")
''','''
P_dn = init()
for _ in range(800):
    noisy = X + rng.normal(0, 0.3, X.shape)
    step(P_dn, noisy, X, lr=2.0)

def reconstruct(P, X_in):
    return np.maximum(0, X_in @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]

test_noisy = X + np.random.default_rng(1).normal(0, 0.3, X.shape)
err_plain = np.mean((reconstruct(P, test_noisy) - X) ** 2)
err_dn = np.mean((reconstruct(P_dn, test_noisy) - X) ** 2)
print(err_plain, err_dn)
fig, axes = plt.subplots(3, 6, figsize=(8, 4))
for j in range(6):
    for i, img in enumerate([test_noisy[j], reconstruct(P, test_noisy[j:j+1])[0], reconstruct(P_dn, test_noisy[j:j+1])[0]]):
        axes[i, j].imshow(img.reshape(8, 8), cmap="gray"); axes[i, j].axis("off")
''','''
assert err_dn < err_plain, "The denoising autoencoder should clean noisy inputs better"
''', "Input noisy, target X: step(P_dn, noisy, X, lr=2.0)"),
]),

dict(id='DL7', title='Regularisation, one tool at a time', packages=['numpy', 'scikit-learn'], intro=
"Overfit a network on purpose, then add regularisers one at a time and measure each. sklearn's MLPClassifier has L2 (alpha) and early stopping built in; you'll write dropout yourself.",
steps=[
S('Overfit on purpose',
"Train a big network on only 150 digit images. Training accuracy hits 100% while test accuracy lags: that gap is overfitting.",
'''
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
X, y = load_digits(return_X_y=True); X = X / 16.0
Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=150, random_state=0)
big = MLPClassifier(hidden_layer_sizes=(256, 256), alpha=0.0, max_iter=500, random_state=0).fit(Xtr, ytr)
gap = big.score(Xtr, ytr) - ___
print(big.score(Xtr, ytr), big.score(Xte, yte), gap)
''','''
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
X, y = load_digits(return_X_y=True); X = X / 16.0
Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=150, random_state=0)
big = MLPClassifier(hidden_layer_sizes=(256, 256), alpha=0.0, max_iter=500, random_state=0).fit(Xtr, ytr)
gap = big.score(Xtr, ytr) - big.score(Xte, yte)
print(big.score(Xtr, ytr), big.score(Xte, yte), gap)
''','''
assert big.score(Xtr, ytr) == 1.0 and gap > 0.05, "Train accuracy 1.0 and a clear gap to test accuracy"
''', "Subtract the test score: big.score(Xte, yte)"),
S('L2 and data augmentation',
"Add L2 (alpha), then augment the training set with noisy copies of each image. Record test accuracy for each. Don't expect every regulariser to help: on this data one of them barely moves the needle. Measuring instead of assuming is the real lesson.",
'''
l2_alpha = ___          # try 0.1, 1.0, 3.0
results = {"none": big.score(Xte, yte)}
results["L2"] = MLPClassifier(hidden_layer_sizes=(256, 256), alpha=l2_alpha, max_iter=500, random_state=0).fit(Xtr, ytr).score(Xte, yte)
rng = np.random.default_rng(0)
X_aug = np.vstack([Xtr] + [np.clip(Xtr + rng.normal(0, 0.15, Xtr.shape), 0, 1) for _ in range(4)])
y_aug = np.concatenate([ytr] * 5)
results["augment"] = MLPClassifier(hidden_layer_sizes=(256, 256), alpha=0.0, max_iter=500, random_state=0).fit(X_aug, y_aug).score(Xte, yte)
print(results)
''','''
l2_alpha = 1.0          # try 0.1, 1.0, 3.0
results = {"none": big.score(Xte, yte)}
results["L2"] = MLPClassifier(hidden_layer_sizes=(256, 256), alpha=l2_alpha, max_iter=500, random_state=0).fit(Xtr, ytr).score(Xte, yte)
rng = np.random.default_rng(0)
X_aug = np.vstack([Xtr] + [np.clip(Xtr + rng.normal(0, 0.15, Xtr.shape), 0, 1) for _ in range(4)])
y_aug = np.concatenate([ytr] * 5)
results["augment"] = MLPClassifier(hidden_layer_sizes=(256, 256), alpha=0.0, max_iter=500, random_state=0).fit(X_aug, y_aug).score(Xte, yte)
print(results)
''','''
assert l2_alpha > 0, "Pick a positive alpha"
assert results["augment"] > results["none"], "Augmentation should beat the unregularised network"
''', "Any positive number, e.g. 1.0. Then compare all three results"),
S('Write dropout yourself',
"Inverted dropout: during training, zero each unit with probability p and divide the survivors by (1 - p) so the expected value is unchanged. At test time, do nothing.",
'''
def dropout(h, p, training, rng):
    if not training:
        return h
    keep = rng.random(h.shape) >= p
    return ___

h = np.ones((10000, 50))
out = dropout(h, 0.5, True, np.random.default_rng(0))
print(out.mean(), (out == 0).mean())
''','''
def dropout(h, p, training, rng):
    if not training:
        return h
    keep = rng.random(h.shape) >= p
    return h * keep / (1 - p)

h = np.ones((10000, 50))
out = dropout(h, 0.5, True, np.random.default_rng(0))
print(out.mean(), (out == 0).mean())
''','''
assert abs(out.mean() - 1) < 0.02, "Dividing by (1 - p) should keep the mean at 1"
assert abs((out == 0).mean() - 0.5) < 0.02, "About half the units should be zeroed"
assert np.array_equal(dropout(h, 0.5, False, None), h), "At test time dropout should do nothing"
''', "h * keep / (1 - p)"),
]),

dict(id='DL8', title='Initialisation and batch norm', packages=['numpy', 'matplotlib'], intro=
"Push random data through a 10-layer network and watch the activations. Bad initialisation makes them vanish or explode; good initialisation (or batch norm) keeps them healthy.",
steps=[
S('Activations through depth',
"For each layer: h = relu(h W). Record the standard deviation of h at every layer for three initialisations: tiny (sd 0.01), Xavier (sd sqrt(1/n)) and He (sd sqrt(2/n)).",
'''
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
n, depth = 256, 10
X = rng.normal(size=(1000, n))

def layer_stds(sd):
    h, stds = X, []
    for _ in range(depth):
        W = rng.normal(0, sd, (n, n))
        h = np.maximum(0, h @ W)
        stds.append(h.std())
    return stds

inits = {"tiny": 0.01, "Xavier": np.sqrt(1 / n), "He": ___}
stds = {k: layer_stds(v) for k, v in inits.items()}
for k, v in stds.items():
    plt.semilogy(range(1, depth + 1), v, "o-", label=k)
plt.legend(); plt.xlabel("layer"); plt.ylabel("std of activations")
print({k: round(v[-1], 6) for k, v in stds.items()})
''','''
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
n, depth = 256, 10
X = rng.normal(size=(1000, n))

def layer_stds(sd):
    h, stds = X, []
    for _ in range(depth):
        W = rng.normal(0, sd, (n, n))
        h = np.maximum(0, h @ W)
        stds.append(h.std())
    return stds

inits = {"tiny": 0.01, "Xavier": np.sqrt(1 / n), "He": np.sqrt(2 / n)}
stds = {k: layer_stds(v) for k, v in inits.items()}
for k, v in stds.items():
    plt.semilogy(range(1, depth + 1), v, "o-", label=k)
plt.legend(); plt.xlabel("layer"); plt.ylabel("std of activations")
print({k: round(v[-1], 6) for k, v in stds.items()})
''','''
assert stds["tiny"][-1] < 1e-6, "Tiny init should make activations vanish"
assert 0.3 < stds["He"][-1] < 3, "He init should keep activations around the same scale"
assert stds["Xavier"][-1] < stds["He"][-1], "With ReLU, Xavier shrinks more than He"
''', "np.sqrt(2 / n)"),
S('Batch norm rescues bad initialisation',
"Batch norm standardises each unit over the batch: (h - mean) / sqrt(var + eps), then applies a learned scale and shift (here gamma = 1, beta = 0). Insert it before every ReLU in the tiny-init network.",
'''
def batchnorm(h, eps=1e-5):
    mu = h.mean(axis=0)
    var = h.var(axis=0)
    return ___

def layer_stds_bn(sd):
    h, out = X, []
    for _ in range(depth):
        W = rng.normal(0, sd, (n, n))
        h = np.maximum(0, batchnorm(h @ W))
        out.append(h.std())
    return out

bn = layer_stds_bn(0.01)
print(np.round(bn, 3))
''','''
def batchnorm(h, eps=1e-5):
    mu = h.mean(axis=0)
    var = h.var(axis=0)
    return (h - mu) / np.sqrt(var + eps)

def layer_stds_bn(sd):
    h, out = X, []
    for _ in range(depth):
        W = rng.normal(0, sd, (n, n))
        h = np.maximum(0, batchnorm(h @ W))
        out.append(h.std())
    return out

bn = layer_stds_bn(0.01)
print(np.round(bn, 3))
''','''
z = batchnorm(rng.normal(5, 3, (500, 4)))
assert np.allclose(z.mean(axis=0), 0, atol=1e-9) and np.allclose(z.std(axis=0), 1, atol=1e-3), "Each column should end up with mean 0, std 1"
assert min(bn) > 0.3, "With batch norm, activations should stay healthy even with tiny init"
''', "(h - mu) / np.sqrt(var + eps)"),
]),

dict(id='DL9', title='word2vec: skip-gram with negative sampling', packages=['numpy'], intro=
"Words that appear in similar contexts get similar vectors. Train skip-gram with negative sampling on a tiny made-up corpus about banking, then probe what it learned.",
steps=[
S('Make (centre, context) training pairs',
"Slide a window of 2 words on each side over every sentence. Each nearby word gives a (centre, context) pair.",
'''
import numpy as np
sentences = [
    "customer deposits cash at branch", "customer withdraws cash at atm", "customer transfers money online",
    "bank approves home loan", "bank rejects car loan", "bank approves personal loan",
    "customer deposits cheque at branch", "customer withdraws money at atm", "bank rejects home loan",
    "customer transfers cash online", "bank approves car loan", "bank rejects personal loan",
] * 30
vocab = sorted({w for s in sentences for w in s.split()})
idx = {w: i for i, w in enumerate(vocab)}
pairs = []
for s in sentences:
    words = s.split()
    for i, w in enumerate(words):
        for j in range(max(0, i - 2), min(len(words), i + 3)):
            if ___:
                pairs.append((idx[w], idx[words[j]]))
pairs = np.array(pairs)
print(len(vocab), len(pairs))
''','''
import numpy as np
sentences = [
    "customer deposits cash at branch", "customer withdraws cash at atm", "customer transfers money online",
    "bank approves home loan", "bank rejects car loan", "bank approves personal loan",
    "customer deposits cheque at branch", "customer withdraws money at atm", "bank rejects home loan",
    "customer transfers cash online", "bank approves car loan", "bank rejects personal loan",
] * 30
vocab = sorted({w for s in sentences for w in s.split()})
idx = {w: i for i, w in enumerate(vocab)}
pairs = []
for s in sentences:
    words = s.split()
    for i, w in enumerate(words):
        for j in range(max(0, i - 2), min(len(words), i + 3)):
            if j != i:
                pairs.append((idx[w], idx[words[j]]))
pairs = np.array(pairs)
print(len(vocab), len(pairs))
''','''
assert len(pairs) == 30 * sum(sum(1 for j in range(max(0, i - 2), min(len(s.split()), i + 3)) if j != i) for s in sentences[:12] for i in range(len(s.split()))), "Skip the word itself: j != i"
''', "A word isn't its own context: j != i"),
S('Train with negative sampling',
"For each true pair, push sigma(u_centre . v_context) toward 1; for 5 random 'negative' words, push it toward 0. The gradient of -log sigma(s) with respect to s is sigma(s) - 1, and for a negative it's sigma(s).",
'''
rng = np.random.default_rng(0)
V, D = len(vocab), 10
U = rng.normal(0, 0.1, (V, D))   # centre-word vectors
Vc = rng.normal(0, 0.1, (V, D))  # context-word vectors
sig = lambda z: 1 / (1 + np.exp(-z))
lr, K = 0.05, 5
for epoch in range(15):
    for c, o in pairs[rng.permutation(len(pairs))]:
        negs = rng.integers(0, V, K)
        u = U[c].copy()
        g_pos = sig(u @ Vc[o]) - 1
        g_neg = ___                      # one number per negative word
        U[c] -= lr * (g_pos * Vc[o] + g_neg @ Vc[negs])
        Vc[o] -= lr * g_pos * u
        Vc[negs] -= lr * np.outer(g_neg, u)

def similar(word, k=3):
    E = U / np.linalg.norm(U, axis=1, keepdims=True)
    sims = E @ E[idx[word]]
    return [vocab[i] for i in np.argsort(-sims) if vocab[i] != word][:k]
print(similar("deposits"), similar("approves"), similar("atm"))
''','''
rng = np.random.default_rng(0)
V, D = len(vocab), 10
U = rng.normal(0, 0.1, (V, D))   # centre-word vectors
Vc = rng.normal(0, 0.1, (V, D))  # context-word vectors
sig = lambda z: 1 / (1 + np.exp(-z))
lr, K = 0.05, 5
for epoch in range(15):
    for c, o in pairs[rng.permutation(len(pairs))]:
        negs = rng.integers(0, V, K)
        u = U[c].copy()
        g_pos = sig(u @ Vc[o]) - 1
        g_neg = sig(Vc[negs] @ u)        # one number per negative word
        U[c] -= lr * (g_pos * Vc[o] + g_neg @ Vc[negs])
        Vc[o] -= lr * g_pos * u
        Vc[negs] -= lr * np.outer(g_neg, u)

def similar(word, k=3):
    E = U / np.linalg.norm(U, axis=1, keepdims=True)
    sims = E @ E[idx[word]]
    return [vocab[i] for i in np.argsort(-sims) if vocab[i] != word][:k]
print(similar("deposits"), similar("approves"), similar("atm"))
''','''
assert "rejects" in similar("approves", 2), "'approves' and 'rejects' share contexts, so they should end up close"
assert "withdraws" in similar("deposits", 3) or "transfers" in similar("deposits", 3), "Customer actions should cluster together"
''', "sig(Vc[negs] @ u): the score of each negative word, through the sigmoid"),
]),

dict(id='DL10', title='Convolution from scratch', packages=['numpy', 'scikit-learn', 'matplotlib'], intro=
"A convolution slides a small filter over an image and takes a dot product at each position. Write it, check the output-size formula, and see what edge filters do. (Training a real CNN on CIFAR-10 needs PyTorch: that part is a guided Colab notebook, linked at the end.)",
steps=[
S('conv2d with stride and padding',
"Output size = (N + 2P - K) / S + 1. Pad the image with zeros, then for each output position take the dot product of the filter with the patch under it.",
'''
import numpy as np
import matplotlib.pyplot as plt

def conv2d(img, k, stride=1, pad=0):
    img = np.pad(img, pad)
    K = k.shape[0]
    out_size = (img.shape[0] - K) // stride + 1
    out = np.zeros((out_size, out_size))
    for i in range(out_size):
        for j in range(out_size):
            patch = img[i * stride:i * stride + K, j * stride:j * stride + K]
            out[i, j] = ___
    return out

img = np.arange(36.0).reshape(6, 6)
print(conv2d(img, np.ones((3, 3)) / 9).shape, conv2d(img, np.ones((3, 3)), stride=2, pad=1).shape)
''','''
import numpy as np
import matplotlib.pyplot as plt

def conv2d(img, k, stride=1, pad=0):
    img = np.pad(img, pad)
    K = k.shape[0]
    out_size = (img.shape[0] - K) // stride + 1
    out = np.zeros((out_size, out_size))
    for i in range(out_size):
        for j in range(out_size):
            patch = img[i * stride:i * stride + K, j * stride:j * stride + K]
            out[i, j] = np.sum(patch * k)
    return out

img = np.arange(36.0).reshape(6, 6)
print(conv2d(img, np.ones((3, 3)) / 9).shape, conv2d(img, np.ones((3, 3)), stride=2, pad=1).shape)
''','''
from scipy.signal import correlate2d
r = np.random.default_rng(0)
a, k = r.normal(size=(7, 7)), r.normal(size=(3, 3))
assert np.allclose(conv2d(a, k), correlate2d(a, k, mode="valid")), "Each output is np.sum(patch * k)"
assert conv2d(np.zeros((32, 32)), np.zeros((5, 5)), stride=1, pad=2).shape == (32, 32), "(32 + 4 - 5)/1 + 1 = 32"
''', "np.sum(patch * k)"),
S('Edge detectors',
"A vertical-edge filter responds where brightness changes left to right. Apply it and its transpose (horizontal edges) to a digit image.",
'''
from sklearn.datasets import load_digits
digit = load_digits().images[0]
big = np.kron(digit, np.ones((4, 4)))          # upscale 8x8 -> 32x32 so it's easier to see
sobel_v = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
sobel_h = ___
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, im, t in zip(axes, [big, conv2d(big, sobel_v, pad=1), conv2d(big, sobel_h, pad=1)], ["digit", "vertical edges", "horizontal edges"]):
    ax.imshow(im, cmap="gray"); ax.set_title(t); ax.axis("off")
''','''
from sklearn.datasets import load_digits
digit = load_digits().images[0]
big = np.kron(digit, np.ones((4, 4)))          # upscale 8x8 -> 32x32 so it's easier to see
sobel_v = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
sobel_h = sobel_v.T
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, im, t in zip(axes, [big, conv2d(big, sobel_v, pad=1), conv2d(big, sobel_h, pad=1)], ["digit", "vertical edges", "horizontal edges"]):
    ax.imshow(im, cmap="gray"); ax.set_title(t); ax.axis("off")
''','''
assert np.array_equal(sobel_h, sobel_v.T), "The horizontal filter is the transpose: sobel_v.T"
''', "sobel_v.T"),
S('Max pooling',
"Pooling shrinks a feature map by keeping the largest value in each 2x2 block, which gives a little translation invariance.",
'''
def maxpool2(x):
    H, W = x.shape
    return x[:H // 2 * 2, :W // 2 * 2].reshape(H // 2, 2, W // 2, 2).___
print(maxpool2(np.arange(16).reshape(4, 4)))
''','''
def maxpool2(x):
    H, W = x.shape
    return x[:H // 2 * 2, :W // 2 * 2].reshape(H // 2, 2, W // 2, 2).max(axis=(1, 3))
print(maxpool2(np.arange(16).reshape(4, 4)))
''','''
assert np.array_equal(maxpool2(np.arange(16).reshape(4, 4)), [[5, 7], [13, 15]])
''', "Take the max over the two 'inside the block' axes: .max(axis=(1, 3))"),
]),

dict(id='DL11', title='RNNs and vanishing gradients', packages=['numpy', 'matplotlib'], intro=
"A recurrent network reuses the same weights at every time step. That's elegant, and it's why gradients vanish over long sequences. See it numerically, then see why LSTMs help. (Training a character-level LSTM is the Colab notebook linked at the end.)",
steps=[
S('An RNN forward pass',
"h_t = tanh(W_h h_(t-1) + W_x x_t + b). Run it over a sequence and keep every hidden state.",
'''
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
H, D, T = 16, 4, 50
W_h = rng.normal(0, 1 / np.sqrt(H), (H, H))
W_x = rng.normal(0, 0.5, (H, D))
b = np.zeros(H)
xs = rng.normal(size=(T, D))

def rnn_forward(xs, W_h):
    h = np.zeros(H); hs = []
    for x in xs:
        h = ___
        hs.append(h)
    return np.array(hs)

hs = rnn_forward(xs, W_h)
print(hs.shape)
''','''
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
H, D, T = 16, 4, 50
W_h = rng.normal(0, 1 / np.sqrt(H), (H, H))
W_x = rng.normal(0, 0.5, (H, D))
b = np.zeros(H)
xs = rng.normal(size=(T, D))

def rnn_forward(xs, W_h):
    h = np.zeros(H); hs = []
    for x in xs:
        h = np.tanh(W_h @ h + W_x @ x + b)
        hs.append(h)
    return np.array(hs)

hs = rnn_forward(xs, W_h)
print(hs.shape)
''','''
assert hs.shape == (50, 16) and np.allclose(hs[0], np.tanh(W_x @ xs[0])), "h_t = tanh(W_h h_(t-1) + W_x x_t + b)"
''', "np.tanh(W_h @ h + W_x @ x + b)"),
S('Measure the gradient through time',
"How much does h_50 depend on h_1? Multiply the step Jacobians: dh_t/dh_(t-1) = diag(1 - h_t^2) W_h. Track the norm of the running product as you go back in time, for small and large W_h.",
'''
def grad_norms(scale):
    Wh = W_h * scale
    hs = rnn_forward(xs, Wh)
    J = np.eye(H); norms = []
    for t in range(T - 1, 0, -1):
        J = J @ ___
        norms.append(np.linalg.norm(J))
    return norms

for scale in [0.5, 1.0, 3.0]:
    plt.semilogy(grad_norms(scale), label=f"W_h x {scale}")
plt.xlabel("steps back in time"); plt.ylabel("||d h_T / d h_t||"); plt.legend()
print(grad_norms(0.5)[-1], grad_norms(1.0)[-1])
''','''
def grad_norms(scale):
    Wh = W_h * scale
    hs = rnn_forward(xs, Wh)
    J = np.eye(H); norms = []
    for t in range(T - 1, 0, -1):
        J = J @ (np.diag(1 - hs[t] ** 2) @ Wh)
        norms.append(np.linalg.norm(J))
    return norms

for scale in [0.5, 1.0, 3.0]:
    plt.semilogy(grad_norms(scale), label=f"W_h x {scale}")
plt.xlabel("steps back in time"); plt.ylabel("||d h_T / d h_t||"); plt.legend()
print(grad_norms(0.5)[-1], grad_norms(1.0)[-1])
''','''
g = grad_norms(0.5)
assert g[-1] < 1e-6 * g[0], "With small W_h the gradient should vanish over 50 steps"
''', "The step Jacobian is np.diag(1 - hs[t] ** 2) @ Wh"),
S("Why an LSTM's cell state helps",
"An LSTM's cell state updates as c_t = f_t * c_(t-1) + i_t * g_t, so dc_t/dc_(t-1) is just the forget gate f_t, with no squashing weight matrix. If the forget gates stay near 1, gradients survive. Compute the gradient after 50 steps for forget gates around 0.97.",
'''
forget = 1 / (1 + np.exp(-rng.normal(3.5, 0.3, T)))   # sigmoid of large values: near 1
lstm_grad = ___                                       # product of forget gates across time
print(forget.mean(), lstm_grad, grad_norms(1.0)[-1])
''','''
forget = 1 / (1 + np.exp(-rng.normal(3.5, 0.3, T)))   # sigmoid of large values: near 1
lstm_grad = np.prod(forget)                           # product of forget gates across time
print(forget.mean(), lstm_grad, grad_norms(1.0)[-1])
''','''
assert abs(lstm_grad - np.prod(forget)) < 1e-12 and lstm_grad > 0.1, "The cell-state gradient is the product of the forget gates"
''', "np.prod(forget)"),
]),

dict(id='DL12', title='Attention from scratch', packages=['numpy', 'matplotlib'], intro=
"Attention lets every position look at every other position and decide what's relevant. It's the core of every transformer, and it's about five lines of NumPy.",
steps=[
S('Scaled dot-product attention',
"Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V. Each row of the weight matrix sums to 1 and says how much each query attends to each key.",
'''
import numpy as np
import matplotlib.pyplot as plt

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def attention(Q, K, V, mask=None):
    d = Q.shape[-1]
    scores = ___
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores)
    return weights @ V, weights

rng = np.random.default_rng(0)
Q, K, V = rng.normal(size=(4, 8)), rng.normal(size=(6, 8)), rng.normal(size=(6, 3))
out, w = attention(Q, K, V)
print(out.shape, w.sum(axis=1))
''','''
import numpy as np
import matplotlib.pyplot as plt

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def attention(Q, K, V, mask=None):
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores)
    return weights @ V, weights

rng = np.random.default_rng(0)
Q, K, V = rng.normal(size=(4, 8)), rng.normal(size=(6, 8)), rng.normal(size=(6, 3))
out, w = attention(Q, K, V)
print(out.shape, w.sum(axis=1))
''','''
assert out.shape == (4, 3) and np.allclose(w.sum(axis=1), 1)
assert np.allclose(w, softmax(Q @ K.T / np.sqrt(8))), "scores = Q @ K.T / sqrt(d)"
''', "Q @ K.T / np.sqrt(d)"),
S('A causal mask',
"In a language model, position t may only look at positions 0..t. Build a lower-triangular True/False mask and check that no weight lands on the future.",
'''
T = 6
X = rng.normal(size=(T, 8))
causal = ___          # True where attention is allowed
out, w = attention(X, X, X, mask=causal)
plt.imshow(w, cmap="viridis"); plt.xlabel("key position"); plt.ylabel("query position"); plt.colorbar()
print(np.round(w, 2))
''','''
T = 6
X = rng.normal(size=(T, 8))
causal = np.tril(np.ones((T, T), dtype=bool))          # True where attention is allowed
out, w = attention(X, X, X, mask=causal)
plt.imshow(w, cmap="viridis"); plt.xlabel("key position"); plt.ylabel("query position"); plt.colorbar()
print(np.round(w, 2))
''','''
assert np.allclose(np.triu(w, 1), 0), "No weight should fall above the diagonal (the future)"
assert np.allclose(w[0], [1, 0, 0, 0, 0, 0]), "The first position can only attend to itself"
''', "np.tril(np.ones((T, T), dtype=bool))"),
S('Attention retrieves by content',
"Store 5 key-value pairs. Query with a noisy copy of key 3: attention should return (almost) value 3. This is why attention works as a soft lookup table, like a JOIN on similarity instead of equality.",
'''
keys = rng.normal(size=(5, 16)) * 3
values = np.eye(5)
query = keys[3] + rng.normal(0, 0.3, 16)
out, w = attention(query[None, :], keys, values)
retrieved = int(___)
print(np.round(w, 3), retrieved)
''','''
keys = rng.normal(size=(5, 16)) * 3
values = np.eye(5)
query = keys[3] + rng.normal(0, 0.3, 16)
out, w = attention(query[None, :], keys, values)
retrieved = int(out.argmax())
print(np.round(w, 3), retrieved)
''','''
assert retrieved == 3 and w[0, 3] > 0.9, "The query should pull out value 3"
''', "out.argmax()"),
]),
]

P07 = dict(id='P07', title='Your own autograd library: train an MLP', packages=['numpy', 'scikit-learn', 'matplotlib'], intro=
"Extend the DL3 engine into a small library: neurons, layers, an MLP and a training loop, then classify the two-moons data. Everything runs on your own Value class.",
steps=[
S('The Value engine (with relu)',
"This is the DL3 class plus relu and power. Fill in relu's backward: the gradient passes through only where the output was positive.",
'''
import math, random
import numpy as np

class Value:
    def __init__(self, data, parents=()):
        self.data, self.grad, self._parents, self._backward = data, 0.0, parents, lambda: None
    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data + o.data, (self, o))
        def bw(): self.grad += out.grad; o.grad += out.grad
        out._backward = bw; return out
    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data * o.data, (self, o))
        def bw(): self.grad += o.data * out.grad; o.grad += self.data * out.grad
        out._backward = bw; return out
    def __pow__(self, k):
        out = Value(self.data ** k, (self,))
        def bw(): self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = bw; return out
    def relu(self):
        out = Value(max(0.0, self.data), (self,))
        def bw(): self.grad += ___
        out._backward = bw; return out
    __radd__ = __add__; __rmul__ = __mul__
    def __neg__(self): return self * -1
    def __sub__(self, o): return self + (-o)
    def backward(self):
        order, seen = [], set()
        def visit(v):
            if v not in seen:
                seen.add(v); [visit(p) for p in v._parents]; order.append(v)
        visit(self); self.grad = 1.0
        for v in reversed(order): v._backward()
''','''
import math, random
import numpy as np

class Value:
    def __init__(self, data, parents=()):
        self.data, self.grad, self._parents, self._backward = data, 0.0, parents, lambda: None
    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data + o.data, (self, o))
        def bw(): self.grad += out.grad; o.grad += out.grad
        out._backward = bw; return out
    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data * o.data, (self, o))
        def bw(): self.grad += o.data * out.grad; o.grad += self.data * out.grad
        out._backward = bw; return out
    def __pow__(self, k):
        out = Value(self.data ** k, (self,))
        def bw(): self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = bw; return out
    def relu(self):
        out = Value(max(0.0, self.data), (self,))
        def bw(): self.grad += (out.data > 0) * out.grad
        out._backward = bw; return out
    __radd__ = __add__; __rmul__ = __mul__
    def __neg__(self): return self * -1
    def __sub__(self, o): return self + (-o)
    def backward(self):
        order, seen = [], set()
        def visit(v):
            if v not in seen:
                seen.add(v); [visit(p) for p in v._parents]; order.append(v)
        visit(self); self.grad = 1.0
        for v in reversed(order): v._backward()
''','''
a = Value(2.0); r = a.relu(); r.backward()
b = Value(-1.0); s = b.relu(); s.backward()
assert a.grad == 1.0 and b.grad == 0.0, "relu passes the gradient only where the output is positive"
''', "(out.data > 0) * out.grad"),
S('Neuron, Layer, MLP',
"A neuron computes w . x + b, optionally through relu. A layer is a list of neurons; an MLP is a list of layers. `parameters()` collects every weight so the optimiser can update them.",
'''
random.seed(0)
class Neuron:
    def __init__(self, n_in, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0); self.nonlin = nonlin
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_in, n_out, **kw):
        self.neurons = [Neuron(n_in, **kw) for _ in range(n_out)]
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    def parameters(self):
        return [p for n in self.neurons for p in ___]

class MLP:
    def __init__(self, n_in, sizes):
        dims = [n_in] + sizes
        self.layers = [Layer(dims[i], dims[i + 1], nonlin=i != len(sizes) - 1) for i in range(len(sizes))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

model = MLP(2, [8, 8, 1])
print(len(model.parameters()))
''','''
random.seed(0)
class Neuron:
    def __init__(self, n_in, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0); self.nonlin = nonlin
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_in, n_out, **kw):
        self.neurons = [Neuron(n_in, **kw) for _ in range(n_out)]
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, n_in, sizes):
        dims = [n_in] + sizes
        self.layers = [Layer(dims[i], dims[i + 1], nonlin=i != len(sizes) - 1) for i in range(len(sizes))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

model = MLP(2, [8, 8, 1])
print(len(model.parameters()))
''','''
assert len(model.parameters()) == (2 * 8 + 8) + (8 * 8 + 8) + (8 + 1), "Collect every neuron's parameters"
''', "n.parameters()"),
S('Train on the two moons',
"Labels are -1/+1. Use the hinge loss mean(relu(1 - y * score)) plus a little L2, and plain SGD. 40 points keeps it manageable, but expect about 30 seconds: pure Python is slow, which is exactly why real libraries use arrays underneath.",
'''
from sklearn.datasets import make_moons
Xm, ym = make_moons(40, noise=0.1, random_state=0)
ym = ym * 2 - 1
for k in range(60):
    scores = [model(list(map(Value, x))) for x in Xm]
    losses = [(1 + -yi * si).relu() for yi, si in zip(ym, scores)]
    loss = sum(losses) * (1.0 / len(losses)) + 1e-4 * sum(p * p for p in model.parameters())
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()
    lr = 1.0 - 0.9 * k / 60
    for p in model.parameters():
        p.data -= ___
acc = np.mean([(s.data > 0) == (yi > 0) for s, yi in zip(scores, ym)])
print(loss.data, acc)
''','''
from sklearn.datasets import make_moons
Xm, ym = make_moons(40, noise=0.1, random_state=0)
ym = ym * 2 - 1
for k in range(60):
    scores = [model(list(map(Value, x))) for x in Xm]
    losses = [(1 + -yi * si).relu() for yi, si in zip(ym, scores)]
    loss = sum(losses) * (1.0 / len(losses)) + 1e-4 * sum(p * p for p in model.parameters())
    for p in model.parameters():
        p.grad = 0.0
    loss.backward()
    lr = 1.0 - 0.9 * k / 60
    for p in model.parameters():
        p.data -= lr * p.grad
acc = np.mean([(s.data > 0) == (yi > 0) for s, yi in zip(scores, ym)])
print(loss.data, acc)
''','''
assert acc >= 0.9, f"Aim for at least 90% training accuracy; you have {acc:.2f}"
''', "lr * p.grad"),
S('Check your gradients against finite differences',
"The real test of an autograd library: nudge one weight, see how the loss changes, and compare with the gradient your engine computed.",
'''
def loss_fn():
    sc = [model(list(map(Value, x))) for x in Xm[:10]]
    return sum((1 + -yi * si).relu() for yi, si in zip(ym[:10], sc)) * 0.1

p = model.parameters()[5]
for q in model.parameters():
    q.grad = 0.0
L = loss_fn(); L.backward()
engine_grad = p.grad
h = 1e-6
p.data += h; up = loss_fn().data
p.data -= 2 * h; down = loss_fn().data
p.data += h
numeric_grad = ___
print(engine_grad, numeric_grad)
''','''
def loss_fn():
    sc = [model(list(map(Value, x))) for x in Xm[:10]]
    return sum((1 + -yi * si).relu() for yi, si in zip(ym[:10], sc)) * 0.1

p = model.parameters()[5]
for q in model.parameters():
    q.grad = 0.0
L = loss_fn(); L.backward()
engine_grad = p.grad
h = 1e-6
p.data += h; up = loss_fn().data
p.data -= 2 * h; down = loss_fn().data
p.data += h
numeric_grad = (up - down) / (2 * h)
print(engine_grad, numeric_grad)
''','''
assert abs(engine_grad - numeric_grad) < 1e-4, "Your engine's gradient should match the numerical estimate"
''', "(up - down) / (2 * h)"),
])

DL_ALL = DL + [P07]
