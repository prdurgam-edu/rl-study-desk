"""Guided labs for phase 1 projects (P02-P04) and Markov chain weekly tasks (MC1-MC9)."""
from labs_warmup import S

P1 = [
dict(id='P02', title='Linear algebra from scratch', packages=['numpy', 'matplotlib', 'scikit-learn'], intro=
"Build the three workhorses of linear algebra in ML yourself: orthonormal bases, least squares, and PCA via SVD. Each one is a few lines once you see it.",
steps=[
S('Gram-Schmidt',
"Gram-Schmidt turns any set of independent columns into orthonormal ones. For each column: subtract its projections onto the columns already done, then divide by its length.\n\nThe check confirms Q's columns are orthonormal (`Q.T @ Q` is the identity).",
'''
import numpy as np

def gram_schmidt(A):
    A = A.astype(float)
    Q = np.zeros_like(A)
    for j in range(A.shape[1]):
        v = A[:, j].copy()
        for i in range(j):
            v = v - (Q[:, i] @ A[:, j]) * Q[:, i]
        Q[:, j] = ___
    return Q

A = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
Q = gram_schmidt(A)
print(np.round(Q.T @ Q, 6))
''','''
import numpy as np

def gram_schmidt(A):
    A = A.astype(float)
    Q = np.zeros_like(A)
    for j in range(A.shape[1]):
        v = A[:, j].copy()
        for i in range(j):
            v = v - (Q[:, i] @ A[:, j]) * Q[:, i]
        Q[:, j] = v / np.linalg.norm(v)
    return Q

A = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
Q = gram_schmidt(A)
print(np.round(Q.T @ Q, 6))
''','''
Qt = gram_schmidt(np.array([[2., 1], [1, 3], [0, 1]]))
assert np.allclose(Q.T @ Q, np.eye(3)), "Q.T @ Q should be the identity matrix"
assert np.allclose(Qt.T @ Qt, np.eye(2)), "Should also work for a 3x2 matrix"
''', "Divide v by its length: v / np.linalg.norm(v)"),
S('Least squares two ways',
"To fit y = w0 + w1 x, build X with a column of ones and a column of x. The normal equations give w = (X^T X)^-1 X^T y. Solve them with `np.linalg.solve` (never invert explicitly), then compare with `np.linalg.lstsq`.",
'''
rng = np.random.default_rng(0)
x = rng.uniform(0, 5, 50)
y = 1.5 + 2.0 * x + rng.normal(0, 0.5, 50)

X = np.column_stack([np.ones_like(x), x])
w_normal = np.linalg.solve(___, ___)
w_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
print(w_normal, w_lstsq)
''','''
rng = np.random.default_rng(0)
x = rng.uniform(0, 5, 50)
y = 1.5 + 2.0 * x + rng.normal(0, 0.5, 50)

X = np.column_stack([np.ones_like(x), x])
w_normal = np.linalg.solve(X.T @ X, X.T @ y)
w_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
print(w_normal, w_lstsq)
''','''
assert np.allclose(w_normal, w_lstsq), "Your normal-equation answer should match lstsq"
assert abs(w_normal[1] - 2) < 0.3, "The slope should be close to 2"
''', "np.linalg.solve(X.T @ X, X.T @ y)"),
S('PCA via SVD on handwritten digits',
"Load 8x8 digit images (64 numbers each). Centre the data, take the SVD, and the squared singular values tell you how much variance each component explains. Find how many components keep 95% of the variance.",
'''
from sklearn.datasets import load_digits
D = load_digits().data
Dc = D - D.mean(axis=0)
U, Svals, Vt = np.linalg.svd(Dc, full_matrices=False)
explained = Svals ** 2 / np.sum(Svals ** 2)
cum = np.cumsum(explained)
k95 = ___   # number of components needed to reach 0.95
print(k95)

import matplotlib.pyplot as plt
plt.plot(cum); plt.axhline(0.95, color="gray"); plt.xlabel("components"); plt.ylabel("variance kept")
''','''
from sklearn.datasets import load_digits
D = load_digits().data
Dc = D - D.mean(axis=0)
U, Svals, Vt = np.linalg.svd(Dc, full_matrices=False)
explained = Svals ** 2 / np.sum(Svals ** 2)
cum = np.cumsum(explained)
k95 = int(np.argmax(cum >= 0.95) + 1)
print(k95)

import matplotlib.pyplot as plt
plt.plot(cum); plt.axhline(0.95, color="gray"); plt.xlabel("components"); plt.ylabel("variance kept")
''','''
ref = int(np.argmax(np.cumsum(explained) >= 0.95) + 1)
assert k95 == ref, f"k95 should be {ref}: the first count where cum >= 0.95 (remember +1, Python counts from 0)"
''', "np.argmax(cum >= 0.95) gives the first index where it's True; add 1 to turn an index into a count"),
S('Compression with truncated SVD',
"Keeping only the top k singular values gives the best rank-k approximation of a matrix. Write `reconstruct(M, k)` and watch the error fall as k grows.",
'''
yy, xx = np.mgrid[0:60, 0:80]
img = np.sin(xx / 6) * np.cos(yy / 9) + (xx + yy) / 140 + 0.05 * rng.normal(size=xx.shape)

def reconstruct(M, k):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return ___

errors = [np.linalg.norm(img - reconstruct(img, k)) for k in [1, 2, 5, 20]]
print(np.round(errors, 3))

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, k in zip(axes, [1, 3, 20]):
    ax.imshow(reconstruct(img, k), cmap="gray"); ax.set_title(f"rank {k}"); ax.axis("off")
''','''
yy, xx = np.mgrid[0:60, 0:80]
img = np.sin(xx / 6) * np.cos(yy / 9) + (xx + yy) / 140 + 0.05 * rng.normal(size=xx.shape)

def reconstruct(M, k):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

errors = [np.linalg.norm(img - reconstruct(img, k)) for k in [1, 2, 5, 20]]
print(np.round(errors, 3))

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, k in zip(axes, [1, 3, 20]):
    ax.imshow(reconstruct(img, k), cmap="gray"); ax.set_title(f"rank {k}"); ax.axis("off")
''','''
R = reconstruct(img, 3)
assert R.shape == img.shape, "The reconstruction must have the same shape as the image"
assert np.linalg.matrix_rank(R) == 3, "reconstruct(img, 3) should have rank 3"
assert all(a >= b for a, b in zip(errors, errors[1:])), "Error should fall as k grows"
''', "U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]"),
]),

dict(id='P03', title='Probability simulation lab', packages=['numpy'], intro=
"Check four classic probability results by simulation. When your simulation and the formula agree, you really understand both.",
steps=[
S('Birthday paradox',
"With 23 people, the chance two share a birthday is about 50%. The exact formula: 1 - (365/365)(364/365)...((365-n+1)/365). Write it, then simulate.",
'''
import numpy as np

def p_exact(n):
    p_no_match = 1.0
    for i in range(n):
        p_no_match *= ___
    return 1 - p_no_match

def p_sim(n, trials=20000, seed=0):
    rng = np.random.default_rng(seed)
    bdays = rng.integers(0, 365, size=(trials, n))
    has_dup = [len(set(row)) < n for row in bdays]
    return np.mean(has_dup)

print(p_exact(23), p_sim(23))
''','''
import numpy as np

def p_exact(n):
    p_no_match = 1.0
    for i in range(n):
        p_no_match *= (365 - i) / 365
    return 1 - p_no_match

def p_sim(n, trials=20000, seed=0):
    rng = np.random.default_rng(seed)
    bdays = rng.integers(0, 365, size=(trials, n))
    has_dup = [len(set(row)) < n for row in bdays]
    return np.mean(has_dup)

print(p_exact(23), p_sim(23))
''','''
assert abs(p_exact(23) - 0.5073) < 0.001, "p_exact(23) should be about 0.507"
assert abs(p_exact(23) - p_sim(23)) < 0.02, "Simulation and formula should agree"
''', "The i-th person must avoid the i birthdays already taken: (365 - i) / 365"),
S('Coupon collector',
"How many random draws to see all n coupon types? Theory says n x (1 + 1/2 + ... + 1/n). Simulate it for n = 10.",
'''
def draws_to_collect(n, rng):
    seen, draws = set(), 0
    while len(seen) < n:
        seen.add(rng.integers(0, n))
        draws += 1
    return draws

rng = np.random.default_rng(1)
sim_mean = np.mean([draws_to_collect(10, rng) for _ in range(5000)])
theory = ___
print(sim_mean, theory)
''','''
def draws_to_collect(n, rng):
    seen, draws = set(), 0
    while len(seen) < n:
        seen.add(rng.integers(0, n))
        draws += 1
    return draws

rng = np.random.default_rng(1)
sim_mean = np.mean([draws_to_collect(10, rng) for _ in range(5000)])
theory = 10 * sum(1 / k for k in range(1, 11))
print(sim_mean, theory)
''','''
assert abs(theory - 29.2897) < 0.01, "theory = 10 * sum(1/k for k in 1..10)"
assert abs(sim_mean - theory) < 1.0, "Simulation should be close to theory"
''', "10 * sum(1 / k for k in range(1, 11))"),
S('Estimate pi by throwing darts',
"Throw points uniformly in the unit square; the fraction inside the quarter circle x^2 + y^2 <= 1 is pi/4.",
'''
rng = np.random.default_rng(2)
pts = rng.random((200000, 2))
inside = ___
pi_est = 4 * inside.mean()
print(pi_est)
''','''
rng = np.random.default_rng(2)
pts = rng.random((200000, 2))
inside = (pts ** 2).sum(axis=1) <= 1
pi_est = 4 * inside.mean()
print(pi_est)
''','''
assert abs(pi_est - np.pi) < 0.02, f"pi_est should be close to 3.1416, you have {pi_est}"
''', "(pts ** 2).sum(axis=1) <= 1"),
S('Empirical tail vs the Hoeffding bound',
"For n fair coin flips, Hoeffding says P(|heads/n - 0.5| >= t) <= 2 exp(-2 n t^2). Compare it with simulation for n = 100, t = 0.1. The bound should be true but loose.",
'''
n, t = 100, 0.1
heads = rng.binomial(n, 0.5, size=100000)
empirical = (np.abs(heads / n - 0.5) >= t).mean()
bound = ___
print(empirical, bound)
''','''
n, t = 100, 0.1
heads = rng.binomial(n, 0.5, size=100000)
empirical = (np.abs(heads / n - 0.5) >= t).mean()
bound = 2 * np.exp(-2 * n * t ** 2)
print(empirical, bound)
''','''
assert abs(bound - 2 * np.exp(-2)) < 1e-9, "bound = 2 * exp(-2 n t^2)"
assert empirical < bound, "The empirical probability should sit below the bound"
''', "2 * np.exp(-2 * n * t ** 2)"),
]),

dict(id='P04', title='Gradient descent lab', packages=['numpy', 'matplotlib'], intro=
"Why does gradient descent sometimes crawl? The answer is the condition number. You'll see it, then fix it with momentum.",
steps=[
S('Gradient descent on a quadratic',
"f(x) = 0.5 x^T A x has gradient A x. With A = diag(1, 10) the bowl is stretched. Run 200 steps from (1, 1) and plot the path.",
'''
import numpy as np
import matplotlib.pyplot as plt

A = np.diag([1.0, 10.0])
x = np.array([1.0, 1.0])
lr = 0.15
path = [x.copy()]
for _ in range(200):
    x = x - lr * ___
    path.append(x.copy())
path = np.array(path)
plt.plot(path[:, 0], path[:, 1], "o-", ms=2); plt.title("GD path"); plt.axis("equal")
print(path[-1])
''','''
import numpy as np
import matplotlib.pyplot as plt

A = np.diag([1.0, 10.0])
x = np.array([1.0, 1.0])
lr = 0.15
path = [x.copy()]
for _ in range(200):
    x = x - lr * (A @ x)
    path.append(x.copy())
path = np.array(path)
plt.plot(path[:, 0], path[:, 1], "o-", ms=2); plt.title("GD path"); plt.axis("equal")
print(path[-1])
''','''
assert np.linalg.norm(path[-1]) < 1e-6, "After 200 steps x should be essentially at the minimum (0, 0)"
''', "The gradient is A @ x"),
S('Condition number slows everything down',
"For A = diag(1, kappa) with the best fixed step 1/kappa, count the steps until ||x|| < 1e-3. The count grows roughly in proportion to kappa.",
'''
def iters_needed(kappa, tol=1e-3):
    A = np.diag([1.0, kappa])
    x = np.array([1.0, 1.0])
    lr = 1 / kappa
    for k in range(1, 100000):
        x = x - lr * (A @ x)
        if ___:
            return k

counts = {kappa: iters_needed(kappa) for kappa in [2, 10, 50]}
print(counts)
''','''
def iters_needed(kappa, tol=1e-3):
    A = np.diag([1.0, kappa])
    x = np.array([1.0, 1.0])
    lr = 1 / kappa
    for k in range(1, 100000):
        x = x - lr * (A @ x)
        if np.linalg.norm(x) < tol:
            return k

counts = {kappa: iters_needed(kappa) for kappa in [2, 10, 50]}
print(counts)
''','''
assert counts[2] < counts[10] < counts[50], f"More stretched bowls should need more steps, you have {counts}"
assert 300 < counts[50] < 400, "With kappa = 50 it should take a few hundred steps"
''', "np.linalg.norm(x) < tol"),
S('Momentum to the rescue',
"Momentum keeps a velocity: v = beta v - lr grad, then x = x + v. It rolls through narrow valleys. With beta = 0.9 it should beat plain GD at kappa = 50.",
'''
def iters_momentum(kappa, beta=0.9, tol=1e-3):
    A = np.diag([1.0, kappa])
    x = np.array([1.0, 1.0]); v = np.zeros(2)
    lr = 1 / kappa
    for k in range(1, 100000):
        v = ___
        x = x + v
        if np.linalg.norm(x) < tol:
            return k

print(iters_needed(50), iters_momentum(50))
''','''
def iters_momentum(kappa, beta=0.9, tol=1e-3):
    A = np.diag([1.0, kappa])
    x = np.array([1.0, 1.0]); v = np.zeros(2)
    lr = 1 / kappa
    for k in range(1, 100000):
        v = beta * v - lr * (A @ x)
        x = x + v
        if np.linalg.norm(x) < tol:
            return k

print(iters_needed(50), iters_momentum(50))
''','''
assert iters_momentum(50) < iters_needed(50) / 2, "Momentum should need well under half the steps"
''', "v = beta * v - lr * (A @ x)"),
]),
]

MC = [
dict(id='MC1', title='Simulating a weather Markov chain', packages=['numpy'], intro=
"A Markov chain is a table of transition probabilities. Simulate one and see the long-run frequencies match theory.",
steps=[
S('The transition matrix',
"Row i of P holds the probabilities of moving from state i (0 sunny, 1 cloudy, 2 rainy) to each state. Every row must sum to 1.",
'''
import numpy as np
P = np.array([[0.7, 0.2, 0.1],
              [0.3, 0.4, 0.3],
              [0.2, 0.3, 0.5]])
rows_ok = ___   # True if every row sums to 1
print(rows_ok)
''','''
import numpy as np
P = np.array([[0.7, 0.2, 0.1],
              [0.3, 0.4, 0.3],
              [0.2, 0.3, 0.5]])
rows_ok = np.allclose(P.sum(axis=1), 1)
print(rows_ok)
''','''
assert rows_ok is True or rows_ok == True, "rows_ok should be np.allclose(P.sum(axis=1), 1)"
''', "np.allclose(P.sum(axis=1), 1)"),
S('Simulate 10,000 days',
"From the current state, pick the next one using that row as probabilities: `rng.choice(3, p=P[state])`.",
'''
rng = np.random.default_rng(0)
state, counts = 0, np.zeros(3)
for t in range(10000):
    state = ___
    counts[state] += 1
empirical = counts / counts.sum()
print(empirical)
''','''
rng = np.random.default_rng(0)
state, counts = 0, np.zeros(3)
for t in range(10000):
    state = rng.choice(3, p=P[state])
    counts[state] += 1
empirical = counts / counts.sum()
print(empirical)
''','''
assert counts.sum() == 10000 and abs(empirical.sum() - 1) < 1e-9
assert empirical[0] > empirical[2], "Sunny should be the most common state"
''', "rng.choice(3, p=P[state])"),
S('Theory: the stationary distribution',
"Start from any distribution pi and repeatedly apply pi = pi @ P. It settles at the stationary distribution, which should match your simulation.",
'''
pi = np.ones(3) / 3
for _ in range(1000):
    pi = ___
print(pi, empirical)
''','''
pi = np.ones(3) / 3
for _ in range(1000):
    pi = pi @ P
print(pi, empirical)
''','''
assert np.allclose(pi @ P, pi), "pi should satisfy pi P = pi"
assert np.abs(pi - empirical).max() < 0.02, "Simulation and theory should agree within 0.02"
''', "pi = pi @ P"),
]),

dict(id='MC2', title="Chapman-Kolmogorov and gambler's ruin", packages=['numpy'], intro=
"n-step transition probabilities are just matrix powers. Check that against simulation.",
steps=[
S("Build the gambler's ruin chain",
"States 0..4 are your money. From 1, 2 or 3 you win or lose 1 with probability 0.5 each. States 0 and 4 are absorbing (you stay there).",
'''
import numpy as np
P = np.zeros((5, 5))
P[0, 0] = 1
P[4, 4] = ___
for i in [1, 2, 3]:
    P[i, i - 1] = 0.5
    P[i, i + 1] = ___
print(P)
''','''
import numpy as np
P = np.zeros((5, 5))
P[0, 0] = 1
P[4, 4] = 1
for i in [1, 2, 3]:
    P[i, i - 1] = 0.5
    P[i, i + 1] = 0.5
print(P)
''','''
assert np.allclose(P.sum(axis=1), 1), "Every row must sum to 1"
assert P[4, 4] == 1 and P[2, 3] == 0.5
''', "Absorbing means P[4, 4] = 1; winning is P[i, i + 1] = 0.5"),
S('n-step probabilities',
"`np.linalg.matrix_power(P, n)[i, j]` is the probability of being in j after n steps starting from i. Starting with 2, what's the chance of having reached 4 after 100 steps?",
'''
P100 = np.linalg.matrix_power(P, ___)
p_win = P100[2, 4]
print(p_win)
''','''
P100 = np.linalg.matrix_power(P, 100)
p_win = P100[2, 4]
print(p_win)
''','''
assert abs(p_win - 0.5) < 1e-6, "From the middle of a fair game, you win with probability 0.5"
''', "matrix_power(P, 100)"),
S('Check with simulation',
"Simulate 20,000 games of 10 steps from state 2 and compare the fraction at 4 with `matrix_power(P, 10)[2, 4]`.",
'''
rng = np.random.default_rng(0)
ends = []
for _ in range(20000):
    s = 2
    for _ in range(10):
        s = rng.choice(5, p=P[s])
    ends.append(s)
sim = np.mean(np.array(ends) == 4)
theory = ___
print(sim, theory)
''','''
rng = np.random.default_rng(0)
ends = []
for _ in range(20000):
    s = 2
    for _ in range(10):
        s = rng.choice(5, p=P[s])
    ends.append(s)
sim = np.mean(np.array(ends) == 4)
theory = np.linalg.matrix_power(P, 10)[2, 4]
print(sim, theory)
''','''
assert abs(theory - np.linalg.matrix_power(P, 10)[2, 4]) < 1e-12
assert abs(sim - theory) < 0.015, "Simulation should be within 0.015 of theory"
''', "np.linalg.matrix_power(P, 10)[2, 4]"),
]),

dict(id='MC3', title='Expected hitting times', packages=['numpy'], intro=
"How long until the game ends? Expected hitting times solve a small linear system.",
steps=[
S('Write the equations',
"Let k_i be the expected steps to absorption from i. Then k_0 = k_4 = 0, and for interior i: k_i = 1 + 0.5 k_(i-1) + 0.5 k_(i+1). Rearranged: k_i - 0.5 k_(i-1) - 0.5 k_(i+1) = 1. Fill in the matrix for the unknowns k_1, k_2, k_3.",
'''
import numpy as np
M = np.array([[1.0, -0.5, 0.0],
              [-0.5, 1.0, ___],
              [0.0, ___, 1.0]])
rhs = np.ones(3)
k = np.linalg.solve(M, rhs)
print(k)
''','''
import numpy as np
M = np.array([[1.0, -0.5, 0.0],
              [-0.5, 1.0, -0.5],
              [0.0, -0.5, 1.0]])
rhs = np.ones(3)
k = np.linalg.solve(M, rhs)
print(k)
''','''
assert np.allclose(k, [3, 4, 3]), f"Expected [3, 4, 3] (the formula i(N - i)), you have {k}"
''', "Each interior row has -0.5 for both neighbours"),
S('Check by simulation',
"Play the game from state 2 until it hits 0 or 4, many times, and average the number of steps.",
'''
rng = np.random.default_rng(0)
lengths = []
for _ in range(20000):
    s, n = 2, 0
    while ___:
        s += rng.choice([-1, 1])
        n += 1
    lengths.append(n)
print(np.mean(lengths))
''','''
rng = np.random.default_rng(0)
lengths = []
for _ in range(20000):
    s, n = 2, 0
    while 0 < s < 4:
        s += rng.choice([-1, 1])
        n += 1
    lengths.append(n)
print(np.mean(lengths))
''','''
assert abs(np.mean(lengths) - 4) < 0.15, "The average should be close to k_2 = 4"
''', "Keep going while 0 < s < 4"),
]),

dict(id='MC4', title='Random walks: recurrent or transient?', packages=['numpy', 'matplotlib'], intro=
"A fair random walk keeps coming back to 0; a biased one drifts away for good. See it happen.",
steps=[
S('Simulate two walks',
"Steps are +1 with probability p, else -1. The position is the running sum of steps.",
'''
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(3)

def walk(p, n=10000):
    steps = np.where(rng.random(n) < p, 1, -1)
    return ___

fair, biased = walk(0.5), walk(0.6)
plt.plot(fair, label="p = 0.5"); plt.plot(biased, label="p = 0.6"); plt.legend()
''','''
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(3)

def walk(p, n=10000):
    steps = np.where(rng.random(n) < p, 1, -1)
    return np.cumsum(steps)

fair, biased = walk(0.5), walk(0.6)
plt.plot(fair, label="p = 0.5"); plt.plot(biased, label="p = 0.6"); plt.legend()
''','''
assert len(fair) == 10000 and set(np.diff(fair)) <= {-1, 1}, "Position is the cumulative sum of the steps"
''', "np.cumsum(steps)"),
S('Count returns to zero',
"Count how many times each walk is back at 0. Average over 200 walks to be sure.",
'''
def returns_to_zero(path):
    return ___

fair_r = np.mean([returns_to_zero(walk(0.5)) for _ in range(200)])
biased_r = np.mean([returns_to_zero(walk(0.6)) for _ in range(200)])
print(fair_r, biased_r)
''','''
def returns_to_zero(path):
    return int((path == 0).sum())

fair_r = np.mean([returns_to_zero(walk(0.5)) for _ in range(200)])
biased_r = np.mean([returns_to_zero(walk(0.6)) for _ in range(200)])
print(fair_r, biased_r)
''','''
assert returns_to_zero(np.array([1, 0, -1, 0])) == 2
assert fair_r > 5 * biased_r, "The fair walk should return to 0 many times more often"
''', "(path == 0).sum()"),
]),

dict(id='MC5', title='Stationary distribution three ways', packages=['numpy'], intro=
"Solve pi P = pi by linear algebra, by iteration, and as an eigenvector. All three must agree.",
steps=[
S('As a linear system',
"pi P = pi means (P^T - I) pi^T = 0. Those equations are redundant, so replace the last one with 'the entries sum to 1'.",
'''
import numpy as np
P = np.array([[0.5, 0.5, 0.0],
              [0.25, 0.5, 0.25],
              [0.0, 0.5, 0.5]])
M = P.T - np.eye(3)
M[-1, :] = ___
rhs = np.array([0, 0, 1.0])
pi_lin = np.linalg.solve(M, rhs)
print(pi_lin)
''','''
import numpy as np
P = np.array([[0.5, 0.5, 0.0],
              [0.25, 0.5, 0.25],
              [0.0, 0.5, 0.5]])
M = P.T - np.eye(3)
M[-1, :] = 1
rhs = np.array([0, 0, 1.0])
pi_lin = np.linalg.solve(M, rhs)
print(pi_lin)
''','''
assert np.allclose(pi_lin, [0.25, 0.5, 0.25]), "Expected [0.25, 0.5, 0.25]"
''', "A row of ones encodes 'sum to 1'"),
S('By power iteration',
"Repeat pi = pi @ P from a uniform start.",
'''
pi_pow = np.ones(3) / 3
for _ in range(500):
    pi_pow = ___
print(pi_pow)
''','''
pi_pow = np.ones(3) / 3
for _ in range(500):
    pi_pow = pi_pow @ P
print(pi_pow)
''','''
assert np.allclose(pi_pow, pi_lin)
''', "pi_pow @ P"),
S('As an eigenvector',
"pi is the left eigenvector of P for eigenvalue 1, i.e. an eigenvector of P^T. Pick it and rescale to sum to 1.",
'''
vals, vecs = np.linalg.eig(P.T)
i = np.argmin(np.abs(vals - 1))
v = np.real(vecs[:, i])
pi_eig = ___
print(pi_eig)
''','''
vals, vecs = np.linalg.eig(P.T)
i = np.argmin(np.abs(vals - 1))
v = np.real(vecs[:, i])
pi_eig = v / v.sum()
print(pi_eig)
''','''
assert np.allclose(pi_eig, pi_lin), "Normalise the eigenvector so it sums to 1"
''', "v / v.sum()"),
]),

dict(id='MC6', title='PageRank is a Markov chain', packages=['numpy'], intro=
"Google's original ranking is the stationary distribution of a random surfer. Build it for six pages.",
steps=[
S('Link matrix to transition matrix',
"links[i] lists pages that page i links to. The surfer follows a random outgoing link, so row i spreads probability 1/len(links[i]) across them.",
'''
import numpy as np
links = {0: [1, 2], 1: [2], 2: [0], 3: [2], 4: [2, 3, 5], 5: [4]}
n = 6
T = np.zeros((n, n))
for i, outs in links.items():
    for j in outs:
        T[i, j] = ___
print(T.sum(axis=1))
''','''
import numpy as np
links = {0: [1, 2], 1: [2], 2: [0], 3: [2], 4: [2, 3, 5], 5: [4]}
n = 6
T = np.zeros((n, n))
for i, outs in links.items():
    for j in outs:
        T[i, j] = 1 / len(outs)
print(T.sum(axis=1))
''','''
assert np.allclose(T.sum(axis=1), 1) and abs(T[4, 3] - 1 / 3) < 1e-12
''', "1 / len(outs)"),
S('Add teleporting and iterate',
"With probability d = 0.85 follow a link, otherwise jump to a random page: G = d T + (1 - d)/n. Iterate rank = rank @ G.",
'''
d = 0.85
G = ___
rank = np.ones(n) / n
for _ in range(200):
    rank = rank @ G
top_page = int(np.argmax(rank))
print(rank.round(3), top_page)
''','''
d = 0.85
G = d * T + (1 - d) / n
rank = np.ones(n) / n
for _ in range(200):
    rank = rank @ G
top_page = int(np.argmax(rank))
print(rank.round(3), top_page)
''','''
assert np.allclose(G.sum(axis=1), 1), "Rows of G must sum to 1"
assert abs(rank.sum() - 1) < 1e-9 and top_page == 2, "Page 2 gets the most links, so it should rank first"
''', "d * T + (1 - d) / n"),
]),

dict(id='MC7', title='Exponential waits and Poisson processes', packages=['numpy'], intro=
"Exponential waiting times forget the past. Stack them and you get a Poisson process.",
steps=[
S('Memorylessness',
"For an exponential wait X, P(X > 5 | X > 2) equals P(X > 3). Check with samples (mean wait 2).",
'''
import numpy as np
rng = np.random.default_rng(0)
X = rng.exponential(scale=2, size=500000)
p_gt3 = (X > 3).mean()
p_cond = ___   # P(X > 5 given X > 2): restrict to X > 2 first
print(p_gt3, p_cond)
''','''
import numpy as np
rng = np.random.default_rng(0)
X = rng.exponential(scale=2, size=500000)
p_gt3 = (X > 3).mean()
p_cond = (X[X > 2] > 5).mean()
print(p_gt3, p_cond)
''','''
assert abs(p_cond - p_gt3) < 0.01, "They should match: the exponential forgets how long you've waited"
''', "(X[X > 2] > 5).mean()"),
S('Build a Poisson process',
"Arrival times are running sums of exponential gaps. With rate 3 per unit time, the count in [0, 10] averages 30, and its variance is also 30.",
'''
def count_in_window(rate, T, rng):
    gaps = rng.exponential(scale=1 / rate, size=int(rate * T * 3) + 20)
    arrivals = ___
    return int((arrivals <= T).sum())

counts = np.array([count_in_window(3, 10, rng) for _ in range(5000)])
print(counts.mean(), counts.var())
''','''
def count_in_window(rate, T, rng):
    gaps = rng.exponential(scale=1 / rate, size=int(rate * T * 3) + 20)
    arrivals = np.cumsum(gaps)
    return int((arrivals <= T).sum())

counts = np.array([count_in_window(3, 10, rng) for _ in range(5000)])
print(counts.mean(), counts.var())
''','''
assert abs(counts.mean() - 30) < 0.5 and abs(counts.var() - 30) < 3, "Mean and variance should both be about 30"
''', "np.cumsum(gaps)"),
]),

dict(id='MC8', title='Thinning and compound Poisson', packages=['numpy'], intro=
"Split a Poisson stream at random and each part is still Poisson. Add random sizes to each event and you get a compound Poisson process, the standard model for insurance claims.",
steps=[
S('Thinning',
"Customers arrive at rate 5 per hour; each is a loan enquiry with probability 0.3. Over 8 hours, loan enquiries should be Poisson with mean 5 x 0.3 x 8 = 12.",
'''
import numpy as np
rng = np.random.default_rng(0)
totals = rng.poisson(5 * 8, size=20000)
loans = rng.binomial(totals, ___)
print(loans.mean(), loans.var())
''','''
import numpy as np
rng = np.random.default_rng(0)
totals = rng.poisson(5 * 8, size=20000)
loans = rng.binomial(totals, 0.3)
print(loans.mean(), loans.var())
''','''
assert abs(loans.mean() - 12) < 0.2 and abs(loans.var() - 12) < 1, "Both mean and variance should be about 12"
''', "Each arrival is a loan enquiry with probability 0.3"),
S('Compound Poisson',
"Claims arrive at rate 2 per day, each with an exponential size of mean 1000. Total claims in 30 days has mean rate x t x E[size] = 60,000.",
'''
def total_claims(rng):
    n = rng.poisson(2 * 30)
    return rng.exponential(1000, size=n).sum()

sims = np.array([total_claims(rng) for _ in range(20000)])
expected = ___
print(sims.mean(), expected)
''','''
def total_claims(rng):
    n = rng.poisson(2 * 30)
    return rng.exponential(1000, size=n).sum()

sims = np.array([total_claims(rng) for _ in range(20000)])
expected = 2 * 30 * 1000
print(sims.mean(), expected)
''','''
assert expected == 60000 and abs(sims.mean() - expected) / expected < 0.02
''', "rate x days x mean size"),
]),

dict(id='MC9', title='From Markov chain to MDP', packages=['numpy'], intro=
"An MDP plus a fixed policy is just a Markov chain. Build that chain for a tiny grid world and compute how long the policy takes to reach the goal. This is policy evaluation, the first idea of reinforcement learning.",
steps=[
S('The policy-induced transition matrix',
"A 3x3 grid, cells numbered 0..8 row by row; goal is cell 8 (absorbing). Policy: move right unless on the right edge, then move down. Each move succeeds with probability 0.8, otherwise you stay put.",
'''
import numpy as np
P = np.zeros((9, 9))
for s in range(9):
    if s == 8:
        P[s, s] = 1
        continue
    r, c = divmod(s, 3)
    nxt = s + 1 if c < 2 else s + 3
    P[s, nxt] = ___
    P[s, s] = ___
print(P.sum(axis=1))
''','''
import numpy as np
P = np.zeros((9, 9))
for s in range(9):
    if s == 8:
        P[s, s] = 1
        continue
    r, c = divmod(s, 3)
    nxt = s + 1 if c < 2 else s + 3
    P[s, nxt] = 0.8
    P[s, s] = 0.2
print(P.sum(axis=1))
''','''
assert np.allclose(P.sum(axis=1), 1) and P[0, 1] == 0.8 and P[2, 5] == 0.8 and P[8, 8] == 1
''', "Success 0.8, stay 0.2"),
S('Expected steps to the goal',
"Expected steps k satisfy k = 1 + Q k for non-goal states, where Q is P without the goal row and column. So (I - Q) k = 1.",
'''
Q = P[:8, :8]
k = np.linalg.solve(___, np.ones(8))
print(k.round(2))
''','''
Q = P[:8, :8]
k = np.linalg.solve(np.eye(8) - Q, np.ones(8))
print(k.round(2))
''','''
assert abs(k[0] - 5) < 1e-9, "From the start you need 4 moves, each taking 1/0.8 = 1.25 steps on average: 5"
''', "np.eye(8) - Q"),
]),
]
