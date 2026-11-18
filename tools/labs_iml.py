"""Guided labs for Intro to ML weekly tasks (IML1-IML11) and projects P05, P06."""
from labs_warmup import S

IML = [
dict(id='IML1', title='Bias and variance, measured', packages=['numpy', 'matplotlib'], intro=
"Bias-variance is easier to feel than to derive. Fit simple and complex models to many noisy datasets and measure both.",
steps=[
S('One dataset, three models',
"The true function is sin(x). Draw 20 noisy points and fit polynomials of degree 1, 3 and 9 with `np.polyfit(x, y, degree)`; evaluate with `np.polyval(coefs, x)`.",
'''
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
x_test = np.linspace(0, 3, 50)

def sample_data(n=20):
    x = rng.uniform(0, 3, n)
    return x, np.sin(x) + rng.normal(0, 0.3, n)

x, y = sample_data()
plt.scatter(x, y, c="k")
for deg in [1, 3, 9]:
    coefs = np.polyfit(x, y, ___)
    plt.plot(x_test, np.polyval(coefs, x_test), label=f"degree {deg}")
plt.plot(x_test, np.sin(x_test), "k--", label="truth"); plt.ylim(-1.5, 2); plt.legend()
''','''
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
x_test = np.linspace(0, 3, 50)

def sample_data(n=20):
    x = rng.uniform(0, 3, n)
    return x, np.sin(x) + rng.normal(0, 0.3, n)

x, y = sample_data()
plt.scatter(x, y, c="k")
for deg in [1, 3, 9]:
    coefs = np.polyfit(x, y, deg)
    plt.plot(x_test, np.polyval(coefs, x_test), label=f"degree {deg}")
plt.plot(x_test, np.sin(x_test), "k--", label="truth"); plt.ylim(-1.5, 2); plt.legend()
''','''
assert len(plt.gca().lines) >= 4, "Plot the three fits and the truth"
''', "Pass deg"),
S('Repeat 200 times and measure',
"For each degree, fit 200 fresh datasets and stack the predictions at x_test. Bias^2 is how far the average prediction is from the truth; variance is how much predictions spread around their own average.",
'''
def bias_variance(deg, reps=200):
    preds = []
    for _ in range(reps):
        x, y = sample_data()
        preds.append(np.polyval(np.polyfit(x, y, deg), x_test))
    preds = np.array(preds)
    mean_pred = preds.mean(axis=0)
    bias2 = np.mean((mean_pred - np.sin(x_test)) ** 2)
    variance = ___
    return bias2, variance

results = {d: bias_variance(d) for d in [1, 3, 9]}
for d, (b, v) in results.items():
    print(f"degree {d}: bias^2 {b:.4f}  variance {v:.4f}")
''','''
def bias_variance(deg, reps=200):
    preds = []
    for _ in range(reps):
        x, y = sample_data()
        preds.append(np.polyval(np.polyfit(x, y, deg), x_test))
    preds = np.array(preds)
    mean_pred = preds.mean(axis=0)
    bias2 = np.mean((mean_pred - np.sin(x_test)) ** 2)
    variance = np.mean(preds.var(axis=0))
    return bias2, variance

results = {d: bias_variance(d) for d in [1, 3, 9]}
for d, (b, v) in results.items():
    print(f"degree {d}: bias^2 {b:.4f}  variance {v:.4f}")
''','''
b1, v1 = results[1]; b3, v3 = results[3]; b9, v9 = results[9]
assert b1 > b3, "Degree 1 should have more bias than degree 3"
assert v9 > v3 > v1, "Variance should grow with degree"
''', "np.mean(preds.var(axis=0)): the spread at each x, averaged"),
]),

dict(id='IML2', title='Linear regression three ways', packages=['numpy', 'scikit-learn', 'matplotlib'], intro=
"Fit the same model with the normal equation, with gradient descent, and with sklearn's ridge and lasso. Data: sklearn's diabetes dataset (10 features, disease progression target).",
steps=[
S('Normal equation',
"Add a column of ones for the intercept and solve (X^T X) w = X^T y.",
'''
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

def add_ones(X):
    return np.column_stack([np.ones(len(X)), X])

w = np.linalg.solve(___, ___)
test_mse = np.mean((add_ones(Xte) @ w - yte) ** 2)
print(test_mse)
''','''
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)

def add_ones(X):
    return np.column_stack([np.ones(len(X)), X])

w = np.linalg.solve(add_ones(Xtr).T @ add_ones(Xtr), add_ones(Xtr).T @ ytr)
test_mse = np.mean((add_ones(Xte) @ w - yte) ** 2)
print(test_mse)
''','''
from sklearn.linear_model import LinearRegression
ref = LinearRegression().fit(Xtr, ytr)
assert np.allclose(w[1:], ref.coef_, rtol=1e-4, atol=1e-4), "Your weights should match sklearn's LinearRegression"
''', "A = add_ones(Xtr); solve(A.T @ A, A.T @ ytr)"),
S('Gradient descent gets the same answer',
"Same model, learned with your W8 loop. The features are already scaled, so a larger learning rate works.",
'''
A = add_ones(Xtr)
w_gd = np.zeros(A.shape[1])
lr = 0.5
for step in range(50000):
    grad = 2 * A.T @ (A @ w_gd - ytr) / len(ytr)
    w_gd = ___
print(np.round(w_gd[:4], 1), np.round(w[:4], 1))
''','''
A = add_ones(Xtr)
w_gd = np.zeros(A.shape[1])
lr = 0.5
for step in range(50000):
    grad = 2 * A.T @ (A @ w_gd - ytr) / len(ytr)
    w_gd = w_gd - lr * grad
print(np.round(w_gd[:4], 1), np.round(w[:4], 1))
''','''
gd_mse = np.mean((add_ones(Xte) @ w_gd - yte) ** 2)
assert abs(gd_mse - test_mse) / test_mse < 0.05, "Gradient descent should reach (nearly) the same test error"
''', "w_gd - lr * grad"),
S('Ridge and lasso',
"Ridge shrinks weights; lasso can set them to exactly zero. Sweep alpha and record test error.",
'''
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt
alphas = [0.001, 0.01, 0.1, 1, 10]
ridge_mse, lasso_mse, lasso_zeros = [], [], []
for a in alphas:
    r = Ridge(alpha=a).fit(Xtr, ytr)
    l = Lasso(alpha=a, max_iter=10000).fit(Xtr, ytr)
    ridge_mse.append(np.mean((r.predict(Xte) - yte) ** 2))
    lasso_mse.append(np.mean((l.predict(Xte) - yte) ** 2))
    lasso_zeros.append(int(___))   # how many lasso coefficients are exactly 0
plt.semilogx(alphas, ridge_mse, "o-", label="ridge"); plt.semilogx(alphas, lasso_mse, "o-", label="lasso"); plt.legend(); plt.xlabel("alpha"); plt.ylabel("test MSE")
print(lasso_zeros)
''','''
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt
alphas = [0.001, 0.01, 0.1, 1, 10]
ridge_mse, lasso_mse, lasso_zeros = [], [], []
for a in alphas:
    r = Ridge(alpha=a).fit(Xtr, ytr)
    l = Lasso(alpha=a, max_iter=10000).fit(Xtr, ytr)
    ridge_mse.append(np.mean((r.predict(Xte) - yte) ** 2))
    lasso_mse.append(np.mean((l.predict(Xte) - yte) ** 2))
    lasso_zeros.append(int((l.coef_ == 0).sum()))
plt.semilogx(alphas, ridge_mse, "o-", label="ridge"); plt.semilogx(alphas, lasso_mse, "o-", label="lasso"); plt.legend(); plt.xlabel("alpha"); plt.ylabel("test MSE")
print(lasso_zeros)
''','''
assert lasso_zeros[-1] > lasso_zeros[0], "Stronger lasso should zero out more coefficients"
''', "(l.coef_ == 0).sum()"),
]),

dict(id='IML3', title='Logistic regression from scratch', packages=['numpy', 'scikit-learn'], intro=
"Classify tumours as malignant or benign (sklearn's breast cancer data) with logistic regression you write yourself, then compare with LDA.",
steps=[
S('Sigmoid and prediction',
"The sigmoid squashes any number into (0, 1): s(z) = 1 / (1 + e^-z). Prediction: probability = s(X w).",
'''
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler().fit(Xtr)
Xtr = np.column_stack([np.ones(len(Xtr)), sc.transform(Xtr)])
Xte = np.column_stack([np.ones(len(Xte)), sc.transform(Xte)])

def sigmoid(z):
    return ___
print(sigmoid(np.array([-100, 0, 100])))
''','''
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler().fit(Xtr)
Xtr = np.column_stack([np.ones(len(Xtr)), sc.transform(Xtr)])
Xte = np.column_stack([np.ones(len(Xte)), sc.transform(Xte)])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
print(sigmoid(np.array([-100, 0, 100])))
''','''
assert np.allclose(sigmoid(np.array([0.0, 2.0])), [0.5, 1 / (1 + np.exp(-2))]), "sigmoid(z) = 1 / (1 + exp(-z))"
''', "1 / (1 + np.exp(-z))"),
S('Cross-entropy loss and its gradient',
"Loss = -mean(y log p + (1 - y) log(1 - p)). Its gradient has a beautifully simple form: X^T (p - y) / n.",
'''
def loss(w):
    p = sigmoid(Xtr @ w)
    return -np.mean(ytr * np.log(p + 1e-12) + (1 - ytr) * np.log(1 - p + 1e-12))

def grad(w):
    p = sigmoid(Xtr @ w)
    return ___
print(loss(np.zeros(Xtr.shape[1])))
''','''
def loss(w):
    p = sigmoid(Xtr @ w)
    return -np.mean(ytr * np.log(p + 1e-12) + (1 - ytr) * np.log(1 - p + 1e-12))

def grad(w):
    p = sigmoid(Xtr @ w)
    return Xtr.T @ (p - ytr) / len(ytr)
print(loss(np.zeros(Xtr.shape[1])))
''','''
w0 = np.random.default_rng(0).normal(0, 0.1, Xtr.shape[1]); h = 1e-6; e = np.zeros_like(w0); e[3] = h
num = (loss(w0 + e) - loss(w0 - e)) / (2 * h)
assert abs(grad(w0)[3] - num) < 1e-5, "Your gradient doesn't match the numerical one"
''', "Xtr.T @ (p - ytr) / len(ytr)"),
S('Train and compare with LDA',
"Run gradient descent, then measure test accuracy. Compare with sklearn's LinearDiscriminantAnalysis.",
'''
w = np.zeros(Xtr.shape[1])
for _ in range(3000):
    w = w - 0.5 * grad(w)
acc = np.mean((sigmoid(Xte @ w) > 0.5) == ___)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda_acc = LinearDiscriminantAnalysis().fit(Xtr[:, 1:], ytr).score(Xte[:, 1:], yte)
print(acc, lda_acc)
''','''
w = np.zeros(Xtr.shape[1])
for _ in range(3000):
    w = w - 0.5 * grad(w)
acc = np.mean((sigmoid(Xte @ w) > 0.5) == yte)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda_acc = LinearDiscriminantAnalysis().fit(Xtr[:, 1:], ytr).score(Xte[:, 1:], yte)
print(acc, lda_acc)
''','''
assert acc > 0.94, f"Expected test accuracy above 0.94, you have {acc:.3f}"
''', "Compare with the true test labels, yte"),
]),

dict(id='IML4', title='Perceptrons and SVMs on the two moons', packages=['numpy', 'scikit-learn', 'matplotlib'], intro=
"Two interleaved half-moons can't be split by a straight line. See how a linear model, a linear SVM and an RBF-kernel SVM cope.",
steps=[
S('Linear models',
"Fit sklearn's Perceptron and a linear SVC and record test accuracy.",
'''
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
X, y = make_moons(400, noise=0.2, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
perc_acc = Perceptron(random_state=0).fit(Xtr, ytr).score(Xte, yte)
lin_acc = SVC(kernel=___).fit(Xtr, ytr).score(Xte, yte)
print(perc_acc, lin_acc)
''','''
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
X, y = make_moons(400, noise=0.2, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
perc_acc = Perceptron(random_state=0).fit(Xtr, ytr).score(Xte, yte)
lin_acc = SVC(kernel="linear").fit(Xtr, ytr).score(Xte, yte)
print(perc_acc, lin_acc)
''','''
assert 0.75 < lin_acc < 0.95, "A linear SVM should do OK but not great here"
''', 'kernel="linear"'),
S('RBF kernel and the C/gamma sweep',
"The RBF kernel lets the boundary curve. C trades margin for training errors; gamma sets how wiggly the boundary can be. Find the best pair on the test set (in real work you'd use cross-validation).",
'''
best = (0, None)
for C in [0.1, 1, 10, 100]:
    for gamma in [0.1, 1, 10]:
        acc = SVC(kernel="rbf", C=C, gamma=gamma).fit(Xtr, ytr).score(Xte, yte)
        if acc > best[0]:
            best = (acc, (C, gamma))
rbf_acc, best_params = ___
print(rbf_acc, best_params)
''','''
best = (0, None)
for C in [0.1, 1, 10, 100]:
    for gamma in [0.1, 1, 10]:
        acc = SVC(kernel="rbf", C=C, gamma=gamma).fit(Xtr, ytr).score(Xte, yte)
        if acc > best[0]:
            best = (acc, (C, gamma))
rbf_acc, best_params = best
print(rbf_acc, best_params)
''','''
assert rbf_acc > lin_acc, "The RBF kernel should beat the linear SVM"
''', "best is already a (accuracy, params) pair"),
S('See the boundaries',
"Plot both decision boundaries on a grid. `model.predict` on every grid point, then `contourf`.",
'''
import matplotlib.pyplot as plt
xx, yy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for ax, model, name in [(axes[0], SVC(kernel="linear"), "linear"), (axes[1], SVC(kernel="rbf", C=best_params[0], gamma=best_params[1]), "rbf")]:
    model.fit(Xtr, ytr)
    Z = model.predict(___).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3); ax.scatter(X[:, 0], X[:, 1], c=y, s=8); ax.set_title(name)
''','''
import matplotlib.pyplot as plt
xx, yy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
for ax, model, name in [(axes[0], SVC(kernel="linear"), "linear"), (axes[1], SVC(kernel="rbf", C=best_params[0], gamma=best_params[1]), "rbf")]:
    model.fit(Xtr, ytr)
    Z = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3); ax.scatter(X[:, 0], X[:, 1], c=y, s=8); ax.set_title(name)
''','''
assert len(plt.gcf().axes) == 2, "Two panels expected"
''', "Predict on grid"),
]),

dict(id='IML5', title='A neural network with hand-written backprop', packages=['numpy', 'scikit-learn', 'matplotlib'], intro=
"A one-hidden-layer network on 8x8 handwritten digits, in plain NumPy. After this, backprop will never be magic again.",
steps=[
S('Data and the forward pass',
"Hidden layer: h = relu(X W1 + b1). Output: probabilities = softmax(h W2 + b2). Softmax turns scores into probabilities: exp(s) / sum(exp(s)).",
'''
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
X, y = load_digits(return_X_y=True)
X = X / 16.0
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
Y = np.eye(10)[ytr]               # one-hot labels
rng = np.random.default_rng(0)
W1 = rng.normal(0, 0.1, (64, 64)); b1 = np.zeros(64)
W2 = rng.normal(0, 0.1, (64, 10)); b2 = np.zeros(10)

def softmax(s):
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return ___

def forward(X):
    h = np.maximum(0, X @ W1 + b1)
    p = softmax(h @ W2 + b2)
    return h, p

h, p = forward(Xtr)
print(p.shape, p[0].sum())
''','''
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
X, y = load_digits(return_X_y=True)
X = X / 16.0
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
Y = np.eye(10)[ytr]               # one-hot labels
rng = np.random.default_rng(0)
W1 = rng.normal(0, 0.1, (64, 64)); b1 = np.zeros(64)
W2 = rng.normal(0, 0.1, (64, 10)); b2 = np.zeros(10)

def softmax(s):
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)

def forward(X):
    h = np.maximum(0, X @ W1 + b1)
    p = softmax(h @ W2 + b2)
    return h, p

h, p = forward(Xtr)
print(p.shape, p[0].sum())
''','''
assert np.allclose(p.sum(axis=1), 1), "Each row of softmax output must sum to 1"
''', "e / e.sum(axis=1, keepdims=True)"),
S('Backward pass',
"With softmax + cross-entropy, the output error is simply d2 = (p - Y) / n. Push it back: dW2 = h^T d2, then through relu: d1 = (d2 W2^T) where h > 0, and dW1 = X^T d1.",
'''
def backward(X, Y, h, p):
    n = len(X)
    d2 = (p - Y) / n
    dW2 = h.T @ d2
    db2 = d2.sum(axis=0)
    d1 = (d2 @ W2.T) * (h > 0)
    dW1 = ___
    db1 = d1.sum(axis=0)
    return dW1, db1, dW2, db2

def ce_loss(X, Y):
    return -np.mean(np.sum(Y * np.log(forward(X)[1] + 1e-12), axis=1))
''','''
def backward(X, Y, h, p):
    n = len(X)
    d2 = (p - Y) / n
    dW2 = h.T @ d2
    db2 = d2.sum(axis=0)
    d1 = (d2 @ W2.T) * (h > 0)
    dW1 = X.T @ d1
    db1 = d1.sum(axis=0)
    return dW1, db1, dW2, db2

def ce_loss(X, Y):
    return -np.mean(np.sum(Y * np.log(forward(X)[1] + 1e-12), axis=1))
''','''
Xs, Ys = Xtr[:20], Y[:20]
hh, pp = forward(Xs)
g = backward(Xs, Ys, hh, pp)[0]
eps = 1e-5; old = W1[5, 7]
W1[5, 7] = old + eps; up = ce_loss(Xs, Ys)
W1[5, 7] = old - eps; dn = ce_loss(Xs, Ys)
W1[5, 7] = old
assert abs(g[5, 7] - (up - dn) / (2 * eps)) < 1e-6, "dW1 doesn't match the numerical gradient: it should be X.T @ d1"
''', "dW1 = X.T @ d1"),
S('Train it',
"Full-batch gradient descent for 1500 steps. Aim for over 95% test accuracy.",
'''
lr = 0.5
for step in range(1500):
    h, p = forward(Xtr)
    dW1, db1, dW2, db2 = backward(Xtr, Y, h, p)
    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * ___; b2 -= lr * db2
test_acc = np.mean(forward(Xte)[1].argmax(axis=1) == yte)
print(test_acc)
''','''
lr = 0.5
for step in range(1500):
    h, p = forward(Xtr)
    dW1, db1, dW2, db2 = backward(Xtr, Y, h, p)
    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2
test_acc = np.mean(forward(Xte)[1].argmax(axis=1) == yte)
print(test_acc)
''','''
assert test_acc > 0.95, f"Aim for > 0.95 test accuracy; you have {test_acc:.3f}"
''', "W2 -= lr * dW2"),
]),

dict(id='IML6', title='A decision tree from scratch', packages=['numpy', 'scikit-learn'], intro=
"A decision tree is a stack of IF/ELSE rules learned from data. Build the splitting logic yourself and compare with sklearn.",
steps=[
S('Gini impurity',
"Gini = 1 - sum of squared class proportions. 0 means a pure node; 0.5 is a 50/50 mix of two classes.",
'''
import numpy as np
def gini(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    return ___
print(gini(np.array([0, 0, 1, 1])), gini(np.array([1, 1, 1])))
''','''
import numpy as np
def gini(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=2) / len(y)
    return 1 - np.sum(p ** 2)
print(gini(np.array([0, 0, 1, 1])), gini(np.array([1, 1, 1])))
''','''
assert abs(gini(np.array([0, 0, 1, 1])) - 0.5) < 1e-12 and gini(np.array([1, 1, 1])) == 0
''', "1 - np.sum(p ** 2)"),
dict(free=True, **S('Find the best split',
"Try each feature and a handful of thresholds (quantiles). The best split minimises the size-weighted Gini of the two children.",
'''
def best_split(X, y):
    best = (np.inf, None, None)
    for f in range(X.shape[1]):
        for t in np.unique(np.quantile(X[:, f], np.linspace(0.05, 0.95, 15))):
            left = X[:, f] <= t
            score = (left.sum() * gini(y[left]) + (~left).sum() * gini(y[~left])) / len(y)
            if score < best[0]:
                best = (score, f, t)
    return best[1], best[2]

toy_X = np.array([[1.0], [2], [3], [10], [11], [12]])
toy_y = np.array([0, 0, 0, 1, 1, 1])
f, t = best_split(toy_X, toy_y)
print(f, t)
''','''
def best_split(X, y):
    best = (np.inf, None, None)
    for f in range(X.shape[1]):
        for t in np.unique(np.quantile(X[:, f], np.linspace(0.05, 0.95, 15))):
            left = X[:, f] <= t
            score = (left.sum() * gini(y[left]) + (~left).sum() * gini(y[~left])) / len(y)
            if score < best[0]:
                best = (score, f, t)
    return best[1], best[2]

toy_X = np.array([[1.0], [2], [3], [10], [11], [12]])
toy_y = np.array([0, 0, 0, 1, 1, 1])
f, t = best_split(toy_X, toy_y)
print(f, t)
''','''
assert f == 0 and 3 <= t < 10, "The toy data should split between 3 and 10"
''', "This step is complete: read it, run it, and make sure you follow the weighted-Gini line.")),
S('Grow the tree recursively',
"A node either becomes a leaf (majority class) or splits and builds two children. Stop at max_depth or when the node is pure.",
'''
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)

def build(X, y, depth, max_depth=3):
    majority = int(np.bincount(y, minlength=2).argmax())
    if depth == max_depth or gini(y) == 0:
        return majority
    f, t = best_split(X, y)
    left = X[:, f] <= t
    if left.all() or (~left).all():
        return majority
    return (f, t, build(X[left], y[left], depth + 1, max_depth), ___)

def predict_one(node, x):
    while isinstance(node, tuple):
        f, t, l, r = node
        node = l if x[f] <= t else r
    return node

tree = build(Xtr, ytr, 0)
my_acc = np.mean([predict_one(tree, x) == yy for x, yy in zip(Xte, yte)])
sk_acc = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xtr, ytr).score(Xte, yte)
print(my_acc, sk_acc)
''','''
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)

def build(X, y, depth, max_depth=3):
    majority = int(np.bincount(y, minlength=2).argmax())
    if depth == max_depth or gini(y) == 0:
        return majority
    f, t = best_split(X, y)
    left = X[:, f] <= t
    if left.all() or (~left).all():
        return majority
    return (f, t, build(X[left], y[left], depth + 1, max_depth), build(X[~left], y[~left], depth + 1, max_depth))

def predict_one(node, x):
    while isinstance(node, tuple):
        f, t, l, r = node
        node = l if x[f] <= t else r
    return node

tree = build(Xtr, ytr, 0)
my_acc = np.mean([predict_one(tree, x) == yy for x, yy in zip(Xte, yte)])
sk_acc = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xtr, ytr).score(Xte, yte)
print(my_acc, sk_acc)
''','''
assert my_acc > 0.88 and abs(my_acc - sk_acc) < 0.05, f"Your tree ({my_acc:.3f}) should be close to sklearn's ({sk_acc:.3f})"
''', "The right child is build(X[~left], y[~left], depth + 1, max_depth)"),
]),

dict(id='IML7', title='Cross-validation, ROC and ensembles', packages=['numpy', 'scikit-learn', 'matplotlib'], intro=
"How sure are you about a model's accuracy? Cross-validation and the bootstrap answer that. Then compare bagging and boosting.",
steps=[
S('5-fold cross-validation',
"`cross_val_score(model, X, y, cv=5)` trains 5 times on 4/5 of the data and scores on the held-out fifth.",
'''
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
X, y = load_breast_cancer(return_X_y=True)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
scores = cross_val_score(model, X, y, cv=___)
print(scores.round(3), scores.mean(), scores.std())
''','''
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
X, y = load_breast_cancer(return_X_y=True)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
scores = cross_val_score(model, X, y, cv=5)
print(scores.round(3), scores.mean(), scores.std())
''','''
assert len(scores) == 5 and scores.mean() > 0.95
''', "cv=5"),
S('Bootstrap confidence interval',
"Resample the test predictions with replacement 1000 times; the 2.5th and 97.5th percentiles of accuracy form a 95% interval.",
'''
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
correct = (model.fit(Xtr, ytr).predict(Xte) == yte).astype(float)
rng = np.random.default_rng(0)
boot = [rng.choice(correct, size=len(correct), replace=True).mean() for _ in range(1000)]
lo, hi = ___
print(correct.mean(), lo, hi)
''','''
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
correct = (model.fit(Xtr, ytr).predict(Xte) == yte).astype(float)
rng = np.random.default_rng(0)
boot = [rng.choice(correct, size=len(correct), replace=True).mean() for _ in range(1000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
print(correct.mean(), lo, hi)
''','''
assert lo < correct.mean() < hi and hi - lo < 0.1
''', "np.percentile(boot, [2.5, 97.5])"),
S('ROC curve and AUC',
"The ROC curve shows true-positive vs false-positive rate across every threshold. AUC summarises it in one number.",
'''
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
probs = model.predict_proba(Xte)[:, 1]
fpr, tpr, _ = roc_curve(yte, probs)
auc = ___
plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("false positive rate"); plt.ylabel("true positive rate"); plt.title(f"AUC {auc:.3f}")
''','''
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
probs = model.predict_proba(Xte)[:, 1]
fpr, tpr, _ = roc_curve(yte, probs)
auc = roc_auc_score(yte, probs)
plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("false positive rate"); plt.ylabel("true positive rate"); plt.title(f"AUC {auc:.3f}")
''','''
assert auc > 0.98
''', "roc_auc_score(yte, probs)"),
S('Bagging vs boosting',
"Bagging averages many trees trained on bootstrap samples (cuts variance). AdaBoost trains small trees one after another, each focusing on the previous mistakes (cuts bias).",
'''
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, random_state=0)
ada = AdaBoostClassifier(n_estimators=100, random_state=0)
bag_cv = cross_val_score(bag, X, y, cv=5).mean()
ada_cv = cross_val_score(___, X, y, cv=5).mean()
single_cv = cross_val_score(DecisionTreeClassifier(random_state=0), X, y, cv=5).mean()
print(single_cv, bag_cv, ada_cv)
''','''
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=50, random_state=0)
ada = AdaBoostClassifier(n_estimators=100, random_state=0)
bag_cv = cross_val_score(bag, X, y, cv=5).mean()
ada_cv = cross_val_score(ada, X, y, cv=5).mean()
single_cv = cross_val_score(DecisionTreeClassifier(random_state=0), X, y, cv=5).mean()
print(single_cv, bag_cv, ada_cv)
''','''
assert bag_cv > single_cv and ada_cv > single_cv, "Both ensembles should beat a single tree"
''', "Pass ada"),
]),

dict(id='IML8', title='Random forests and gradient boosting', packages=['numpy', 'scikit-learn', 'matplotlib'], intro=
"The two models that win most tabular problems. (The weekly task mentions the UCI bank-marketing data; this lab uses the built-in breast-cancer data so it runs here. Repeat it on bank data later in your own notebook.)",
steps=[
S('Train both and compare AUC',
"Fit a RandomForestClassifier and a GradientBoostingClassifier; compare test AUC.",
'''
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
data = load_breast_cancer()
Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.3, random_state=0)
rf = RandomForestClassifier(n_estimators=300, random_state=0).fit(Xtr, ytr)
gb = GradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
rf_auc = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])
gb_auc = roc_auc_score(yte, ___)
print(rf_auc, gb_auc)
''','''
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
data = load_breast_cancer()
Xtr, Xte, ytr, yte = train_test_split(data.data, data.target, test_size=0.3, random_state=0)
rf = RandomForestClassifier(n_estimators=300, random_state=0).fit(Xtr, ytr)
gb = GradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
rf_auc = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])
gb_auc = roc_auc_score(yte, gb.predict_proba(Xte)[:, 1])
print(rf_auc, gb_auc)
''','''
assert rf_auc > 0.97 and gb_auc > 0.97
''', "gb.predict_proba(Xte)[:, 1]"),
S('Which features matter?',
"`feature_importances_` shows how much each feature reduced impurity. List the top 5 by name.",
'''
import matplotlib.pyplot as plt
order = np.argsort(rf.feature_importances_)[::-1]
top5 = [data.feature_names[i] for i in ___]
plt.barh(top5[::-1], rf.feature_importances_[order[:5]][::-1])
print(top5)
''','''
import matplotlib.pyplot as plt
order = np.argsort(rf.feature_importances_)[::-1]
top5 = [data.feature_names[i] for i in order[:5]]
plt.barh(top5[::-1], rf.feature_importances_[order[:5]][::-1])
print(top5)
''','''
assert len(top5) == 5 and top5[0] == data.feature_names[np.argmax(rf.feature_importances_)]
''', "order[:5]"),
]),

dict(id='IML9', title='Hidden Markov models: forward and Viterbi', packages=['numpy'], intro=
"The dishonest casino: a dealer secretly switches between a fair die and a loaded one. You see only the rolls. The forward algorithm scores a sequence; Viterbi guesses which die was used when.",
steps=[
S('The model',
"States: 0 fair, 1 loaded. The loaded die rolls a six half the time. Fill in the loaded die's emission probabilities (faces 1-6).",
'''
import numpy as np
from itertools import product
start = np.array([0.5, 0.5])
trans = np.array([[0.95, 0.05],
                  [0.10, 0.90]])
emit = np.array([[1/6] * 6,
                 [0.1, 0.1, 0.1, 0.1, 0.1, ___]])
print(emit.sum(axis=1))
''','''
import numpy as np
from itertools import product
start = np.array([0.5, 0.5])
trans = np.array([[0.95, 0.05],
                  [0.10, 0.90]])
emit = np.array([[1/6] * 6,
                 [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]])
print(emit.sum(axis=1))
''','''
assert np.allclose(emit.sum(axis=1), 1) and emit[1, 5] == 0.5
''', "The sixth face gets 0.5"),
S('Forward algorithm',
"alpha_t(s) = P(rolls so far, state s at time t). Recursion: alpha_(t+1) = (alpha_t @ trans) * emit[:, roll]. The sequence probability is the sum of the last alpha. The check compares with brute force over every hidden path.",
'''
def forward(obs):
    alpha = start * emit[:, obs[0]]
    for o in obs[1:]:
        alpha = ___
    return alpha.sum()

obs = [5, 5, 2, 5, 5, 5, 0]   # faces are 0-based: 5 means a six
print(forward(obs))
''','''
def forward(obs):
    alpha = start * emit[:, obs[0]]
    for o in obs[1:]:
        alpha = (alpha @ trans) * emit[:, o]
    return alpha.sum()

obs = [5, 5, 2, 5, 5, 5, 0]   # faces are 0-based: 5 means a six
print(forward(obs))
''','''
def brute(obs):
    tot = 0
    for path in product([0, 1], repeat=len(obs)):
        p = start[path[0]] * emit[path[0], obs[0]]
        for t in range(1, len(obs)):
            p *= trans[path[t - 1], path[t]] * emit[path[t], obs[t]]
        tot += p
    return tot
assert abs(forward(obs) - brute(obs)) < 1e-15, "forward() should equal the brute-force sum"
''', "(alpha @ trans) * emit[:, o]"),
S('Viterbi: the most likely hidden path',
"Same recursion, but take the max instead of the sum and remember where each max came from. Then trace back.",
'''
def viterbi(obs):
    delta = np.log(start) + np.log(emit[:, obs[0]])
    back = []
    for o in obs[1:]:
        scores = delta[:, None] + np.log(trans)      # scores[i, j]: come from i, go to j
        back.append(scores.argmax(axis=0))
        delta = ___ + np.log(emit[:, o])
    path = [int(delta.argmax())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    return path[::-1]

print(viterbi(obs))
''','''
def viterbi(obs):
    delta = np.log(start) + np.log(emit[:, obs[0]])
    back = []
    for o in obs[1:]:
        scores = delta[:, None] + np.log(trans)      # scores[i, j]: come from i, go to j
        back.append(scores.argmax(axis=0))
        delta = scores.max(axis=0) + np.log(emit[:, o])
    path = [int(delta.argmax())]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    return path[::-1]

print(viterbi(obs))
''','''
def brute_best(obs):
    best, arg = -1, None
    for path in product([0, 1], repeat=len(obs)):
        p = start[path[0]] * emit[path[0], obs[0]]
        for t in range(1, len(obs)):
            p *= trans[path[t - 1], path[t]] * emit[path[t], obs[t]]
        if p > best:
            best, arg = p, list(path)
    return arg
assert viterbi(obs) == brute_best(obs), "Viterbi should match the brute-force best path"
''', "scores.max(axis=0)"),
]),

dict(id='IML10', title='k-means from scratch, and when it fails', packages=['numpy', 'scikit-learn', 'matplotlib'], intro=
"k-means alternates two steps: assign each point to its nearest centre, then move each centre to the mean of its points. It assumes round clusters, which you'll break on purpose.",
steps=[
S('Implement k-means',
"Distances from every point to every centre: `((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)`.",
'''
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score

def kmeans(X, k, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        C = np.array([___ for j in range(k)])
    return labels, C

Xb, yb = make_blobs(300, centers=3, random_state=1)
labels, C = kmeans(Xb, 3)
print(adjusted_rand_score(yb, labels))
''','''
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score

def kmeans(X, k, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        C = np.array([X[labels == j].mean(axis=0) for j in range(k)])
    return labels, C

Xb, yb = make_blobs(300, centers=3, random_state=1)
labels, C = kmeans(Xb, 3)
print(adjusted_rand_score(yb, labels))
''','''
assert adjusted_rand_score(yb, labels) > 0.95, "On round blobs k-means should recover the clusters"
''', "The mean of the points assigned to centre j: X[labels == j].mean(axis=0)"),
S('Moons: k-means fails, DBSCAN succeeds',
"DBSCAN groups points that are densely connected, so it can follow curved shapes. Compare both on the two moons.",
'''
import matplotlib.pyplot as plt
Xm, ym = make_moons(300, noise=0.05, random_state=0)
km_labels, _ = kmeans(Xm, 2)
db_labels = DBSCAN(eps=___).fit_predict(Xm)
km_ari, db_ari = adjusted_rand_score(ym, km_labels), adjusted_rand_score(ym, db_labels)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
axes[0].scatter(Xm[:, 0], Xm[:, 1], c=km_labels, s=8); axes[0].set_title(f"k-means ARI {km_ari:.2f}")
axes[1].scatter(Xm[:, 0], Xm[:, 1], c=db_labels, s=8); axes[1].set_title(f"DBSCAN ARI {db_ari:.2f}")
''','''
import matplotlib.pyplot as plt
Xm, ym = make_moons(300, noise=0.05, random_state=0)
km_labels, _ = kmeans(Xm, 2)
db_labels = DBSCAN(eps=0.2).fit_predict(Xm)
km_ari, db_ari = adjusted_rand_score(ym, km_labels), adjusted_rand_score(ym, db_labels)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
axes[0].scatter(Xm[:, 0], Xm[:, 1], c=km_labels, s=8); axes[0].set_title(f"k-means ARI {km_ari:.2f}")
axes[1].scatter(Xm[:, 0], Xm[:, 1], c=db_labels, s=8); axes[1].set_title(f"DBSCAN ARI {db_ari:.2f}")
''','''
assert db_ari > 0.95 and km_ari < 0.6, f"Try eps between 0.15 and 0.3; DBSCAN ARI is {db_ari:.2f}"
''', "eps=0.2 works well"),
]),

dict(id='IML11', title='EM for a Gaussian mixture', packages=['numpy', 'matplotlib'], intro=
"Expectation-Maximisation fits a mixture of two Gaussians to unlabelled data. E-step: how much each point belongs to each component. M-step: refit each component using those weights.",
steps=[
S('Data and the E-step',
"Responsibility r[i, k] = w_k N(x_i | mu_k, sd_k) / sum over k of the same.",
'''
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(-2, 1, 300), rng.normal(3, 0.7, 200)])

def normal_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

def e_step(x, w, mu, sd):
    dens = np.stack([w[k] * normal_pdf(x, mu[k], sd[k]) for k in range(2)], axis=1)
    return ___

r = e_step(x, np.array([0.5, 0.5]), np.array([-1.0, 1.0]), np.array([1.0, 1.0]))
print(r[:3], r.sum(axis=1)[:3])
''','''
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(-2, 1, 300), rng.normal(3, 0.7, 200)])

def normal_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

def e_step(x, w, mu, sd):
    dens = np.stack([w[k] * normal_pdf(x, mu[k], sd[k]) for k in range(2)], axis=1)
    return dens / dens.sum(axis=1, keepdims=True)

r = e_step(x, np.array([0.5, 0.5]), np.array([-1.0, 1.0]), np.array([1.0, 1.0]))
print(r[:3], r.sum(axis=1)[:3])
''','''
assert r.shape == (500, 2) and np.allclose(r.sum(axis=1), 1), "Each row of responsibilities must sum to 1"
''', "dens / dens.sum(axis=1, keepdims=True)"),
S('M-step and the loop',
"Each component's new weight is its average responsibility; its new mean is the responsibility-weighted mean. Track the log-likelihood: EM guarantees it never goes down.",
'''
w, mu, sd = np.array([0.5, 0.5]), np.array([-1.0, 1.0]), np.array([1.0, 1.0])
loglik = []
for it in range(100):
    r = e_step(x, w, mu, sd)
    Nk = r.sum(axis=0)
    w = Nk / len(x)
    mu = ___
    sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(axis=0) / Nk)
    loglik.append(np.log(sum(w[k] * normal_pdf(x, mu[k], sd[k]) for k in range(2))).sum())
plt.plot(loglik); plt.xlabel("iteration"); plt.ylabel("log-likelihood")
print(w.round(2), mu.round(2), sd.round(2))
''','''
w, mu, sd = np.array([0.5, 0.5]), np.array([-1.0, 1.0]), np.array([1.0, 1.0])
loglik = []
for it in range(100):
    r = e_step(x, w, mu, sd)
    Nk = r.sum(axis=0)
    w = Nk / len(x)
    mu = (r * x[:, None]).sum(axis=0) / Nk
    sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(axis=0) / Nk)
    loglik.append(np.log(sum(w[k] * normal_pdf(x, mu[k], sd[k]) for k in range(2))).sum())
plt.plot(loglik); plt.xlabel("iteration"); plt.ylabel("log-likelihood")
print(w.round(2), mu.round(2), sd.round(2))
''','''
assert all(b >= a - 1e-8 for a, b in zip(loglik, loglik[1:])), "Log-likelihood must never decrease"
assert np.allclose(sorted(mu), [-2, 3], atol=0.2), f"Means should be near -2 and 3, you have {mu}"
''', "(r * x[:, None]).sum(axis=0) / Nk"),
]),
]

PROJ = [
dict(id='P05', title='Your own mini ML library', packages=['numpy', 'scikit-learn'], intro=
"Package what you built in the Intro to ML labs into classes with sklearn's fit/predict interface, and test each against sklearn. This is how real libraries are organised.",
steps=[
S('LinearRegression',
"A class keeps learned parameters on `self`. `fit` learns them and returns `self`; `predict` uses them.",
'''
import numpy as np
from sklearn import linear_model, datasets

class MyLinearRegression:
    def fit(self, X, y):
        A = np.column_stack([np.ones(len(X)), X])
        w = np.linalg.solve(A.T @ A, A.T @ y)
        self.intercept_, self.coef_ = w[0], w[1:]
        return self

    def predict(self, X):
        return ___

Xd, yd = datasets.load_diabetes(return_X_y=True)
mine = MyLinearRegression().fit(Xd, yd)
print(mine.predict(Xd[:3]))
''','''
import numpy as np
from sklearn import linear_model, datasets

class MyLinearRegression:
    def fit(self, X, y):
        A = np.column_stack([np.ones(len(X)), X])
        w = np.linalg.solve(A.T @ A, A.T @ y)
        self.intercept_, self.coef_ = w[0], w[1:]
        return self

    def predict(self, X):
        return self.intercept_ + X @ self.coef_

Xd, yd = datasets.load_diabetes(return_X_y=True)
mine = MyLinearRegression().fit(Xd, yd)
print(mine.predict(Xd[:3]))
''','''
ref = linear_model.LinearRegression().fit(Xd, yd)
assert np.allclose(mine.predict(Xd), ref.predict(Xd)), "Predictions should match sklearn"
''', "self.intercept_ + X @ self.coef_"),
S('LogisticRegression',
"Reuse your IML3 gradient descent inside `fit`. `predict_proba` returns sigmoid scores; `predict` thresholds at 0.5.",
'''
class MyLogisticRegression:
    def __init__(self, lr=0.5, steps=3000):
        self.lr, self.steps = lr, steps

    def fit(self, X, y):
        A = np.column_stack([np.ones(len(X)), X])
        self.w = np.zeros(A.shape[1])
        for _ in range(self.steps):
            p = 1 / (1 + np.exp(-A @ self.w))
            self.w -= self.lr * A.T @ (p - y) / len(y)
        return self

    def predict_proba(self, X):
        A = np.column_stack([np.ones(len(X)), X])
        return 1 / (1 + np.exp(-A @ self.w))

    def predict(self, X):
        return ___

from sklearn.preprocessing import StandardScaler
Xc, yc = datasets.load_breast_cancer(return_X_y=True)
Xc = StandardScaler().fit_transform(Xc)
clf = MyLogisticRegression().fit(Xc, yc)
print((clf.predict(Xc) == yc).mean())
''','''
class MyLogisticRegression:
    def __init__(self, lr=0.5, steps=3000):
        self.lr, self.steps = lr, steps

    def fit(self, X, y):
        A = np.column_stack([np.ones(len(X)), X])
        self.w = np.zeros(A.shape[1])
        for _ in range(self.steps):
            p = 1 / (1 + np.exp(-A @ self.w))
            self.w -= self.lr * A.T @ (p - y) / len(y)
        return self

    def predict_proba(self, X):
        A = np.column_stack([np.ones(len(X)), X])
        return 1 / (1 + np.exp(-A @ self.w))

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

from sklearn.preprocessing import StandardScaler
Xc, yc = datasets.load_breast_cancer(return_X_y=True)
Xc = StandardScaler().fit_transform(Xc)
clf = MyLogisticRegression().fit(Xc, yc)
print((clf.predict(Xc) == yc).mean())
''','''
assert set(np.unique(clf.predict(Xc))) <= {0, 1} and (clf.predict(Xc) == yc).mean() > 0.97
''', "(self.predict_proba(X) > 0.5).astype(int)"),
S('KMeans',
"Wrap your IML10 function as a class with `fit` storing `cluster_centers_` and `labels_`, and `predict` assigning new points to the nearest centre.",
'''
class MyKMeans:
    def __init__(self, k, iters=100, seed=0):
        self.k, self.iters, self.seed = k, iters, seed

    def _assign(self, X):
        d = ((X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
        return d.argmin(axis=1)

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        self.cluster_centers_ = X[rng.choice(len(X), self.k, replace=False)]
        for _ in range(self.iters):
            self.labels_ = self._assign(X)
            self.cluster_centers_ = np.array([X[self.labels_ == j].mean(axis=0) for j in range(self.k)])
        return self

    def predict(self, X):
        return ___

from sklearn.metrics import adjusted_rand_score
Xb, yb = datasets.make_blobs(300, centers=3, random_state=1)
km = MyKMeans(3).fit(Xb)
print(adjusted_rand_score(yb, km.predict(Xb)))
''','''
class MyKMeans:
    def __init__(self, k, iters=100, seed=0):
        self.k, self.iters, self.seed = k, iters, seed

    def _assign(self, X):
        d = ((X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
        return d.argmin(axis=1)

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        self.cluster_centers_ = X[rng.choice(len(X), self.k, replace=False)]
        for _ in range(self.iters):
            self.labels_ = self._assign(X)
            self.cluster_centers_ = np.array([X[self.labels_ == j].mean(axis=0) for j in range(self.k)])
        return self

    def predict(self, X):
        return self._assign(X)

from sklearn.metrics import adjusted_rand_score
Xb, yb = datasets.make_blobs(300, centers=3, random_state=1)
km = MyKMeans(3).fit(Xb)
print(adjusted_rand_score(yb, km.predict(Xb)))
''','''
assert adjusted_rand_score(yb, km.predict(Xb)) > 0.95
''', "self._assign(X)"),
S('A tiny test suite',
"Real libraries ship tests. Write `run_tests()` returning a dict of test name to True/False comparing each class with sklearn. All must pass.",
'''
def run_tests():
    results = {}
    results["linreg"] = np.allclose(MyLinearRegression().fit(Xd, yd).predict(Xd), linear_model.LinearRegression().fit(Xd, yd).predict(Xd))
    sk_acc = linear_model.LogisticRegression(max_iter=1000).fit(Xc, yc).score(Xc, yc)
    results["logreg"] = abs((MyLogisticRegression().fit(Xc, yc).predict(Xc) == yc).mean() - sk_acc) < 0.02
    results["kmeans"] = adjusted_rand_score(yb, MyKMeans(3).fit(Xb).predict(Xb)) > ___
    return results

print(run_tests())
''','''
def run_tests():
    results = {}
    results["linreg"] = np.allclose(MyLinearRegression().fit(Xd, yd).predict(Xd), linear_model.LinearRegression().fit(Xd, yd).predict(Xd))
    sk_acc = linear_model.LogisticRegression(max_iter=1000).fit(Xc, yc).score(Xc, yc)
    results["logreg"] = abs((MyLogisticRegression().fit(Xc, yc).predict(Xc) == yc).mean() - sk_acc) < 0.02
    results["kmeans"] = adjusted_rand_score(yb, MyKMeans(3).fit(Xb).predict(Xb)) > 0.95
    return results

print(run_tests())
''','''
res = run_tests()
assert set(res) == {"linreg", "logreg", "kmeans"} and all(res.values()), f"All tests should pass: {res}"
''', "Use 0.95 as the k-means threshold"),
]),

dict(id='P06', title='End-to-end tabular project', packages=['numpy', 'pandas', 'scikit-learn', 'matplotlib'], intro=
"A realistic workflow: explore, build a clean pipeline, compare models with cross-validation, and pick a decision threshold from business costs. (The project brief mentions bank-marketing data; this lab uses the built-in breast-cancer data so it runs in your browser. The steps carry over one-to-one.)",
steps=[
S('Explore with pandas',
"Put the data in a DataFrame and check the class balance, which you should always do before modelling.",
'''
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer(as_frame=True)
df = data.frame
print(df.shape)
class_balance = df["target"].___()
print(class_balance)
print(df.describe().T.head(5))
''','''
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer(as_frame=True)
df = data.frame
print(df.shape)
class_balance = df["target"].value_counts()
print(class_balance)
print(df.describe().T.head(5))
''','''
assert class_balance.to_dict() == {1: 357, 0: 212}, "Use value_counts() on the target column"
''', "value_counts"),
S('A pipeline and cross-validated comparison',
"A Pipeline bundles preprocessing with the model so nothing leaks from test data into training. Compare logistic regression with gradient boosting by 5-fold ROC AUC.",
'''
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
X, y = df.drop(columns="target"), df["target"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
logit = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000))])
gbm = Pipeline([("model", GradientBoostingClassifier(random_state=0))])
logit_auc = cross_val_score(logit, Xtr, ytr, cv=5, scoring=___).mean()
gbm_auc = cross_val_score(gbm, Xtr, ytr, cv=5, scoring="roc_auc").mean()
print(logit_auc, gbm_auc)
''','''
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
X, y = df.drop(columns="target"), df["target"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
logit = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000))])
gbm = Pipeline([("model", GradientBoostingClassifier(random_state=0))])
logit_auc = cross_val_score(logit, Xtr, ytr, cv=5, scoring="roc_auc").mean()
gbm_auc = cross_val_score(gbm, Xtr, ytr, cv=5, scoring="roc_auc").mean()
print(logit_auc, gbm_auc)
''','''
assert logit_auc > 0.98 and gbm_auc > 0.97
''', 'scoring="roc_auc"'),
S('Choose a threshold from costs',
"Here class 0 is malignant. Missing a malignant case (predicting 1 when the truth is 0) costs 10; a false alarm costs 1. Find the threshold on P(class 0) that minimises total cost on the test set.",
'''
best_model = logit.fit(Xtr, ytr)
p_malignant = best_model.predict_proba(Xte)[:, 0]
def total_cost(th):
    flag = p_malignant >= th              # flagged as malignant
    missed = ((~flag) & (yte.values == 0)).sum()
    false_alarm = (flag & (yte.values == 1)).sum()
    return ___

ths = np.linspace(0.05, 0.95, 19)
costs = [total_cost(t) for t in ths]
best_th = ths[int(np.argmin(costs))]
print(best_th, min(costs), total_cost(0.5))
''','''
best_model = logit.fit(Xtr, ytr)
p_malignant = best_model.predict_proba(Xte)[:, 0]
def total_cost(th):
    flag = p_malignant >= th              # flagged as malignant
    missed = ((~flag) & (yte.values == 0)).sum()
    false_alarm = (flag & (yte.values == 1)).sum()
    return 10 * missed + 1 * false_alarm

ths = np.linspace(0.05, 0.95, 19)
costs = [total_cost(t) for t in ths]
best_th = ths[int(np.argmin(costs))]
print(best_th, min(costs), total_cost(0.5))
''','''
assert min(costs) <= total_cost(0.5), "The best threshold can't be worse than 0.5"
assert best_th <= 0.5, "With misses 10x as costly, you should flag more aggressively (threshold at or below 0.5)"
''', "10 * missed + 1 * false_alarm"),
S('Write the one-paragraph summary',
"Write the result for a manager as a Python string: which model, its cross-validated AUC, the chosen threshold, and the cost saving vs 0.5. The check only looks for those numbers being mentioned.",
'''
summary = f"""
___
"""
print(summary)
''','''
summary = f"""
Logistic regression (cross-validated AUC {logit_auc:.3f}) slightly beat gradient boosting ({gbm_auc:.3f}).
Flagging cases when P(malignant) >= {best_th:.2f} instead of 0.5 cuts total cost from {total_cost(0.5)} to {min(costs)}
on the held-out set, because missed cancers are weighted 10x false alarms.
"""
print(summary)
''','''
assert "___" not in summary and len(summary.split()) >= 25, "Write at least two sentences"
assert f"{best_th:.2f}" in summary, "Mention the chosen threshold (formatted like f'{best_th:.2f}')"
''', "Use an f-string so the numbers fill themselves in: {logit_auc:.3f}, {best_th:.2f}, {min(costs)}"),
]),
]
