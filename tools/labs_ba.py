"""Guided browser labs for Bandit Algorithms weekly tasks BA1-BA12 and project P16."""
from labs_warmup import S
from labs_dl import F

BA = [
dict(id='BA1', title='The halving algorithm', packages=['numpy'], intro=
"Online learning starts here: predict, see the answer, update. If the true rule is in your hypothesis class, predicting with the majority of rules still consistent with the data makes at most log2(class size) mistakes.",
steps=[
S('Majority vote over the version space',
"Hypotheses are thresholds: h_t(x) = 1 if x >= t, for t = 0..99. The truth is t* = 37. Keep the set of thresholds that haven't been wrong yet and predict what most of them say.",
'''
import numpy as np
rng = np.random.default_rng(0)
thresholds = np.arange(100)
t_star = 37
version_space = thresholds.copy()
mistakes = 0
for _ in range(300):
    x = rng.integers(0, 100)
    votes = (x >= version_space)
    pred = int(votes.mean() >= 0.5)
    truth = int(x >= t_star)
    mistakes += pred != truth
    version_space = version_space[___]       # keep thresholds that got this x right
print(mistakes, len(version_space), np.log2(100))
''','''
import numpy as np
rng = np.random.default_rng(0)
thresholds = np.arange(100)
t_star = 37
version_space = thresholds.copy()
mistakes = 0
for _ in range(300):
    x = rng.integers(0, 100)
    votes = (x >= version_space)
    pred = int(votes.mean() >= 0.5)
    truth = int(x >= t_star)
    mistakes += pred != truth
    version_space = version_space[votes == truth]       # keep thresholds that got this x right
print(mistakes, len(version_space), np.log2(100))
''','''
assert t_star in version_space, "The true threshold must never be removed"
assert mistakes <= np.log2(100), "Halving makes at most log2(|H|) mistakes"
''', "Keep the hypotheses whose vote equals the truth: votes == truth"),
S('Why log2?',
"Every mistake means the majority was wrong, so at least half the version space gets thrown away. Count how the version space shrinks after each mistake in a fresh run.",
'''
rng = np.random.default_rng(1)
vs, sizes = thresholds.copy(), [100]
for _ in range(300):
    x = rng.integers(0, 100)
    votes = x >= vs
    pred, truth = int(votes.mean() >= 0.5), int(x >= t_star)
    if pred != truth:
        vs = vs[votes == truth]
        sizes.append(len(vs))
    else:
        vs = vs[votes == truth]
ratios = ___       # each size divided by the previous one
print(sizes, ratios)
''','''
rng = np.random.default_rng(1)
vs, sizes = thresholds.copy(), [100]
for _ in range(300):
    x = rng.integers(0, 100)
    votes = x >= vs
    pred, truth = int(votes.mean() >= 0.5), int(x >= t_star)
    if pred != truth:
        vs = vs[votes == truth]
        sizes.append(len(vs))
    else:
        vs = vs[votes == truth]
ratios = [b / a for a, b in zip(sizes, sizes[1:])]       # each size divided by the previous one
print(sizes, ratios)
''','''
assert all(r <= 0.5 for r in ratios), "After each mistake at most half the version space survives"
''', "[b / a for a, b in zip(sizes, sizes[1:])]"),
]),

dict(id='BA2', title='Weighted majority and Hedge', packages=['numpy', 'matplotlib'], intro=
"Now no expert is perfect. Weighted Majority halves the weight of every expert that errs. Its randomised cousin (Hedge) samples an expert in proportion to weight and gets regret that grows only like sqrt(T log N).",
steps=[
S('Deterministic Weighted Majority',
"Ten experts predict a 0/1 sequence; expert 3 is right 90% of the time, the others 60%. Predict the weighted vote; multiply the weight of each wrong expert by beta = 0.5. Theory: mistakes <= 2.41 (m* + log2 N), where m* is the best expert's mistakes.",
'''
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
T, N = 1000, 10
truth = rng.integers(0, 2, T)
acc = np.full(N, 0.6); acc[3] = 0.9
correct = rng.random((T, N)) < acc
preds = np.where(correct, truth[:, None], 1 - truth[:, None])

w, mistakes = np.ones(N), 0
for t in range(T):
    vote = int(w @ preds[t] >= w.sum() / 2)
    mistakes += vote != truth[t]
    wrong = preds[t] != truth[t]
    w[wrong] *= ___
best_mistakes = (preds != truth[:, None]).sum(axis=0).min()
print(mistakes, best_mistakes, 2.41 * (best_mistakes + np.log2(N)))
''','''
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
T, N = 1000, 10
truth = rng.integers(0, 2, T)
acc = np.full(N, 0.6); acc[3] = 0.9
correct = rng.random((T, N)) < acc
preds = np.where(correct, truth[:, None], 1 - truth[:, None])

w, mistakes = np.ones(N), 0
for t in range(T):
    vote = int(w @ preds[t] >= w.sum() / 2)
    mistakes += vote != truth[t]
    wrong = preds[t] != truth[t]
    w[wrong] *= 0.5
best_mistakes = (preds != truth[:, None]).sum(axis=0).min()
print(mistakes, best_mistakes, 2.41 * (best_mistakes + np.log2(N)))
''','''
assert mistakes <= 2.41 * (best_mistakes + np.log2(N)), "Weighted Majority's mistake bound should hold"
assert np.argmax(w) == 3, "The best expert should end with the largest weight"
''', "Halve them: 0.5"),
S('Hedge and its regret',
"Hedge keeps weights exp(-eta x cumulative loss) and suffers the weighted-average loss. With eta = sqrt(8 ln N / T), regret against the best expert is at most sqrt(T ln N / 2).",
'''
losses = (preds != truth[:, None]).astype(float)
eta = np.sqrt(8 * np.log(N) / T)
cum = np.zeros(N); hedge_loss = 0.0; regret_curve = []
for t in range(T):
    p = np.exp(-eta * cum); p /= p.sum()
    hedge_loss += ___
    cum += losses[t]
    regret_curve.append(hedge_loss - cum.min())
regret = regret_curve[-1]
plt.plot(regret_curve, label="Hedge regret"); plt.plot(np.sqrt(np.arange(1, T + 1) * np.log(N) / 2), label="bound sqrt(t ln N / 2)"); plt.legend(); plt.xlabel("t")
print(regret, np.sqrt(T * np.log(N) / 2))
''','''
losses = (preds != truth[:, None]).astype(float)
eta = np.sqrt(8 * np.log(N) / T)
cum = np.zeros(N); hedge_loss = 0.0; regret_curve = []
for t in range(T):
    p = np.exp(-eta * cum); p /= p.sum()
    hedge_loss += p @ losses[t]
    cum += losses[t]
    regret_curve.append(hedge_loss - cum.min())
regret = regret_curve[-1]
plt.plot(regret_curve, label="Hedge regret"); plt.plot(np.sqrt(np.arange(1, T + 1) * np.log(N) / 2), label="bound sqrt(t ln N / 2)"); plt.legend(); plt.xlabel("t")
print(regret, np.sqrt(T * np.log(N) / 2))
''','''
assert regret <= np.sqrt(T * np.log(N) / 2), "Hedge's regret should respect the bound"
''', "The expected loss this round: p @ losses[t]"),
]),

dict(id='BA3', title='Exp3: bandit feedback against an adversary', packages=['numpy', 'matplotlib'], intro=
"In the bandit setting you see only the reward of the arm you pulled. Exp3 fixes Hedge for this by estimating every arm's reward with importance weighting: the observed reward divided by the probability of having pulled it.",
steps=[
S('Importance-weighted estimates are unbiased',
"If arm i is pulled with probability p_i and pays r_i, the estimate r_hat_i = r_i / p_i when pulled (0 otherwise) has expectation r_i. Check it by simulation.",
'''
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
p = np.array([0.7, 0.2, 0.1]); r = np.array([0.3, 0.8, 0.5])
est = np.zeros(3); n = 200000
for _ in range(n):
    i = rng.choice(3, p=p)
    est[i] += ___
print(est / n, r)
''','''
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
p = np.array([0.7, 0.2, 0.1]); r = np.array([0.3, 0.8, 0.5])
est = np.zeros(3); n = 200000
for _ in range(n):
    i = rng.choice(3, p=p)
    est[i] += r[i] / p[i]
print(est / n, r)
''','''
assert np.allclose(est / n, r, atol=0.02), "Dividing by the pull probability makes the estimate unbiased"
''', "r[i] / p[i]"),
S('Exp3',
"Five arms whose rewards change halfway through (an 'adversary'). Exp3 mixes Hedge-style weights with a little uniform exploration gamma, and feeds Hedge the importance-weighted estimate.",
'''
K, T = 5, 5000
means = np.where(np.arange(T)[:, None] < T // 2, [0.5, 0.6, 0.4, 0.3, 0.2], [0.5, 0.3, 0.4, 0.3, 0.7])
rewards = (rng.random((T, K)) < means).astype(float)
gamma = min(1, np.sqrt(K * np.log(K) / ((np.e - 1) * T)))
w = np.zeros(K)        # log-weights, for numerical safety
got = 0.0
for t in range(T):
    p = np.exp(w - w.max()); p = (1 - gamma) * p / p.sum() + gamma / K
    i = rng.choice(K, p=p)
    got += rewards[t, i]
    r_hat = rewards[t, i] / p[i]
    w[i] += ___
best_fixed = rewards.sum(axis=0).max()
regret = best_fixed - got
print(got, best_fixed, regret, 2.63 * np.sqrt(T * K * np.log(K)))
''','''
K, T = 5, 5000
means = np.where(np.arange(T)[:, None] < T // 2, [0.5, 0.6, 0.4, 0.3, 0.2], [0.5, 0.3, 0.4, 0.3, 0.7])
rewards = (rng.random((T, K)) < means).astype(float)
gamma = min(1, np.sqrt(K * np.log(K) / ((np.e - 1) * T)))
w = np.zeros(K)        # log-weights, for numerical safety
got = 0.0
for t in range(T):
    p = np.exp(w - w.max()); p = (1 - gamma) * p / p.sum() + gamma / K
    i = rng.choice(K, p=p)
    got += rewards[t, i]
    r_hat = rewards[t, i] / p[i]
    w[i] += gamma * r_hat / K
best_fixed = rewards.sum(axis=0).max()
regret = best_fixed - got
print(got, best_fixed, regret, 2.63 * np.sqrt(T * K * np.log(K)))
''','''
assert regret < 2.63 * np.sqrt(T * K * np.log(K)), "Exp3's regret should be within its bound"
assert got > rewards.mean(axis=1).sum(), "Exp3 should beat pulling arms uniformly at random"
''', "The classic Exp3 step: gamma * r_hat / K"),
]),

dict(id='BA4', title='Online gradient descent, and where Follow-the-Leader fails', packages=['numpy', 'matplotlib'], intro=
"Follow-the-Leader (play whatever was best so far) sounds sensible, but an adversary can make it wrong every single round. Online Gradient Descent moves only a little each round and is safe.",
steps=[
S('The classic FTL failure',
"Play x in [-1, 1]; the loss each round is z_t x. Losses are z = 0.5, then -1, +1, -1, +1, ... FTL picks the x minimising the total loss so far, which is always the worst choice for the next round.",
'''
import numpy as np
import matplotlib.pyplot as plt
T = 1000
z = np.array([0.5] + [(-1) ** t for t in range(1, T)], dtype=float)
cum, ftl_loss = 0.0, 0.0
for t in range(T):
    x = 0.0 if t == 0 else -np.sign(cum)        # FTL: minimise cum * x over [-1, 1]
    ftl_loss += z[t] * x
    cum += ___
best_fixed = min(z.sum() * x for x in [-1, 1])
print(ftl_loss, best_fixed)
''','''
import numpy as np
import matplotlib.pyplot as plt
T = 1000
z = np.array([0.5] + [(-1) ** t for t in range(1, T)], dtype=float)
cum, ftl_loss = 0.0, 0.0
for t in range(T):
    x = 0.0 if t == 0 else -np.sign(cum)        # FTL: minimise cum * x over [-1, 1]
    ftl_loss += z[t] * x
    cum += z[t]
best_fixed = min(z.sum() * x for x in [-1, 1])
print(ftl_loss, best_fixed)
''','''
assert ftl_loss - best_fixed > 0.9 * T, "FTL should suffer regret that grows linearly with T"
''', "Accumulate this round's loss coefficient: z[t]"),
S('Online gradient descent',
"OGD steps against the gradient (here just z_t) with step size 1/sqrt(t), then projects back into [-1, 1]. Its regret grows like sqrt(T).",
'''
x, ogd_loss = 0.0, 0.0
for t in range(T):
    ogd_loss += z[t] * x
    x = np.clip(___, -1, 1)
print(ogd_loss - best_fixed, np.sqrt(T))
''','''
x, ogd_loss = 0.0, 0.0
for t in range(T):
    ogd_loss += z[t] * x
    x = np.clip(x - z[t] / np.sqrt(t + 1), -1, 1)
print(ogd_loss - best_fixed, np.sqrt(T))
''','''
assert ogd_loss - best_fixed < 3 * np.sqrt(T), "OGD's regret should be on the order of sqrt(T)"
''', "x - z[t] / np.sqrt(t + 1)"),
]),

dict(id='BA5', title='FTRL: regularise the leader', packages=['numpy'], intro=
"Follow-the-Regularised-Leader adds a regulariser to FTL so the choice can't jump around. With the entropy regulariser on the probability simplex it becomes exactly Hedge; with the squared norm it becomes projected gradient descent.",
steps=[
S('Entropic FTRL in closed form',
"FTRL with regulariser (1/eta) sum x log x over the simplex has the solution x_i proportional to exp(-eta L_i), where L is the cumulative loss vector. Write it.",
'''
import numpy as np
rng = np.random.default_rng(0)
N, T, eta = 4, 200, 0.1
loss_seq = rng.random((T, N))

def ftrl_entropic(L):
    v = np.exp(-eta * (L - L.min()))
    return ___

L = np.zeros(N); w_hedge = np.ones(N) / N; max_diff = 0
for t in range(T):
    x = ftrl_entropic(L)
    max_diff = max(max_diff, np.abs(x - w_hedge).max())
    L += loss_seq[t]
    w_hedge = w_hedge * np.exp(-eta * loss_seq[t]); w_hedge /= w_hedge.sum()
print(max_diff)
''','''
import numpy as np
rng = np.random.default_rng(0)
N, T, eta = 4, 200, 0.1
loss_seq = rng.random((T, N))

def ftrl_entropic(L):
    v = np.exp(-eta * (L - L.min()))
    return v / v.sum()

L = np.zeros(N); w_hedge = np.ones(N) / N; max_diff = 0
for t in range(T):
    x = ftrl_entropic(L)
    max_diff = max(max_diff, np.abs(x - w_hedge).max())
    L += loss_seq[t]
    w_hedge = w_hedge * np.exp(-eta * loss_seq[t]); w_hedge /= w_hedge.sum()
print(max_diff)
''','''
assert max_diff < 1e-12, "Entropic FTRL and multiplicative Hedge should produce identical weights"
''', "Normalise to sum to 1: v / v.sum()"),
S('Euclidean FTRL is projected gradient descent',
"With the regulariser (1/2 eta) ||x||^2, FTRL plays the Euclidean projection of -eta L onto the simplex. The projection routine is given; compare the regret of both regularisers.",
'''
def project_simplex(v):
    u = np.sort(v)[::-1]; css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    tau = (css[rho] - 1) / (rho + 1)
    return np.maximum(v - tau, 0)

def regret(play):
    L, total = np.zeros(N), 0.0
    for t in range(T):
        x = play(L)
        total += x @ loss_seq[t]
        L += loss_seq[t]
    return total - L.min()

reg_entropy = regret(ftrl_entropic)
reg_euclid = regret(lambda L: project_simplex(___))
print(reg_entropy, reg_euclid, regret(lambda L: np.ones(N) / N))
''','''
def project_simplex(v):
    u = np.sort(v)[::-1]; css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    tau = (css[rho] - 1) / (rho + 1)
    return np.maximum(v - tau, 0)

def regret(play):
    L, total = np.zeros(N), 0.0
    for t in range(T):
        x = play(L)
        total += x @ loss_seq[t]
        L += loss_seq[t]
    return total - L.min()

reg_entropy = regret(ftrl_entropic)
reg_euclid = regret(lambda L: project_simplex(-eta * L))
print(reg_entropy, reg_euclid, regret(lambda L: np.ones(N) / N))
''','''
uniform = regret(lambda L: np.ones(N) / N)
assert reg_entropy < uniform and reg_euclid < uniform, "Both FTRL variants should beat playing uniformly"
''', "project_simplex(-eta * L)"),
]),

dict(id='BA6', title='Explore-then-commit', packages=['numpy', 'matplotlib'], intro=
"The simplest stochastic bandit strategy: try each arm m times, then commit to the best-looking one forever. Too little exploration risks committing to the wrong arm; too much wastes pulls. Find the sweet spot.",
steps=[
S('ETC regret for one m',
"Two Gaussian arms with means 0 and -0.5 (gap 0.5), horizon n = 2000. After 2m exploration pulls, commit to the arm with the higher sample mean. Regret = gap x number of pulls of the worse arm.",
'''
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
gap, n = 0.5, 2000

def etc_regret(m, runs=2000):
    means0 = rng.normal(0, 1, (runs, m)).mean(axis=1)
    means1 = rng.normal(-gap, 1, (runs, m)).mean(axis=1)
    wrong = means1 > means0
    return gap * (m + ___ * (n - 2 * m))

print(etc_regret(5), etc_regret(50))
''','''
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
gap, n = 0.5, 2000

def etc_regret(m, runs=2000):
    means0 = rng.normal(0, 1, (runs, m)).mean(axis=1)
    means1 = rng.normal(-gap, 1, (runs, m)).mean(axis=1)
    wrong = means1 > means0
    return gap * (m + wrong.mean() * (n - 2 * m))

print(etc_regret(5), etc_regret(50))
''','''
assert etc_regret(50) < etc_regret(5), "With m = 5 you commit to the wrong arm too often"
''', "The probability of committing wrongly: wrong.mean()"),
S('Sweep m and compare with theory',
"Theory (Lattimore & Szepesvari, Ch. 6) suggests m close to (4 / gap^2) ln(n gap^2 / 4). Sweep m and see where regret bottoms out.",
'''
ms = [2, 5, 10, 20, 30, 50, 80, 120, 200, 400]
regrets = [etc_regret(m) for m in ms]
m_theory = ___
best_m = ms[int(np.argmin(regrets))]
plt.plot(ms, regrets, "o-"); plt.axvline(m_theory, color="gray", ls="--"); plt.xscale("log"); plt.xlabel("m"); plt.ylabel("expected regret")
print(best_m, round(m_theory, 1))
''','''
ms = [2, 5, 10, 20, 30, 50, 80, 120, 200, 400]
regrets = [etc_regret(m) for m in ms]
m_theory = 4 / gap ** 2 * np.log(n * gap ** 2 / 4)
best_m = ms[int(np.argmin(regrets))]
plt.plot(ms, regrets, "o-"); plt.axvline(m_theory, color="gray", ls="--"); plt.xscale("log"); plt.xlabel("m"); plt.ylabel("expected regret")
print(best_m, round(m_theory, 1))
''','''
assert abs(m_theory - 4 / gap ** 2 * np.log(n * gap ** 2 / 4)) < 1e-9
assert regrets[0] > min(regrets) and regrets[-1] > min(regrets), "Regret should be U-shaped in m"
''', "4 / gap ** 2 * np.log(n * gap ** 2 / 4)"),
]),

dict(id='BA7', title='UCB1 vs epsilon-greedy on Bernoulli arms', packages=['numpy', 'matplotlib'], intro=
"UCB plays the arm with the highest optimistic estimate: mean + sqrt(2 ln t / n). Its regret grows logarithmically, while fixed-epsilon greedy's grows linearly. Measure both, and expect a surprise: UCB1 explores so cautiously that it can trail epsilon-greedy for thousands of steps before the curves cross.",
steps=[
S('A vectorised bandit runner',
"Ten Bernoulli arms per run, 200 runs, 10,000 steps. The runner tracks pulls and sums; you supply a `select` function. Regret adds the gap of each arm pulled.",
'''
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
runs, K, T = 200, 10, 10000
mu = rng.uniform(0.1, 0.9, (runs, K))
gaps = mu.max(axis=1, keepdims=True) - mu
R = np.arange(runs)

def run(select):
    N, Ssum = np.zeros((runs, K)), np.zeros((runs, K))
    regret = np.zeros(T)
    for t in range(1, T + 1):
        a = select(N, Ssum, t)
        r = (rng.random(runs) < mu[R, a]).astype(float)
        N[R, a] += 1; Ssum[R, a] += r
        regret[t - 1] = gaps[R, a].mean()
    return np.cumsum(regret)

def ucb1(N, Ssum, t):
    means = Ssum / np.maximum(N, 1)
    index = means + ___
    return np.where(N == 0, np.inf, index).argmax(axis=1)

def eps_greedy(N, Ssum, t, eps=0.1):
    greedy = (Ssum / np.maximum(N, 1)).argmax(axis=1)
    return np.where(rng.random(runs) < eps, rng.integers(0, K, runs), greedy)

reg_ucb, reg_eps = run(ucb1), run(eps_greedy)
plt.plot(reg_ucb, label="UCB1"); plt.plot(reg_eps, label="eps-greedy 0.1"); plt.xlabel("t"); plt.ylabel("cumulative regret"); plt.legend()
print(reg_ucb[-1], reg_eps[-1])
''','''
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
runs, K, T = 200, 10, 10000
mu = rng.uniform(0.1, 0.9, (runs, K))
gaps = mu.max(axis=1, keepdims=True) - mu
R = np.arange(runs)

def run(select):
    N, Ssum = np.zeros((runs, K)), np.zeros((runs, K))
    regret = np.zeros(T)
    for t in range(1, T + 1):
        a = select(N, Ssum, t)
        r = (rng.random(runs) < mu[R, a]).astype(float)
        N[R, a] += 1; Ssum[R, a] += r
        regret[t - 1] = gaps[R, a].mean()
    return np.cumsum(regret)

def ucb1(N, Ssum, t):
    means = Ssum / np.maximum(N, 1)
    index = means + np.sqrt(2 * np.log(t) / np.maximum(N, 1))
    return np.where(N == 0, np.inf, index).argmax(axis=1)

def eps_greedy(N, Ssum, t, eps=0.1):
    greedy = (Ssum / np.maximum(N, 1)).argmax(axis=1)
    return np.where(rng.random(runs) < eps, rng.integers(0, K, runs), greedy)

reg_ucb, reg_eps = run(ucb1), run(eps_greedy)
plt.plot(reg_ucb, label="UCB1"); plt.plot(reg_eps, label="eps-greedy 0.1"); plt.xlabel("t"); plt.ylabel("cumulative regret"); plt.legend()
print(reg_ucb[-1], reg_eps[-1])
''','''
late_slope_ucb = (reg_ucb[-1] - reg_ucb[-1001]) / 1000
late_slope_eps = (reg_eps[-1] - reg_eps[-1001]) / 1000
assert late_slope_ucb < 0.6 * late_slope_eps, "UCB's regret curve should be flattening (logarithmic) while eps-greedy's keeps climbing at a constant rate"
''', "np.sqrt(2 * np.log(t) / np.maximum(N, 1))"),
]),

dict(id='BA8', title='KL-UCB and Thompson sampling', packages=['numpy', 'matplotlib'], intro=
"For 0/1 rewards you can be smarter than UCB1. KL-UCB builds its confidence bound from the Bernoulli KL divergence; Thompson sampling draws from a Beta posterior. Both are asymptotically optimal.",
steps=[
S('The Bernoulli KL divergence and the KL-UCB index',
"kl(p, q) = p ln(p/q) + (1-p) ln((1-p)/(1-q)). KL-UCB's index is the largest q >= p_hat with N kl(p_hat, q) <= ln t, found by bisection.",
'''
import numpy as np
import matplotlib.pyplot as plt
def kl(p, q, eps=1e-12):
    p = np.clip(p, eps, 1 - eps); q = np.clip(q, eps, 1 - eps)
    return ___

def klucb_index(p_hat, n, t, iters=25):
    lo, hi = p_hat.copy(), np.ones_like(p_hat)
    budget = np.log(max(t, 2)) / np.maximum(n, 1)
    for _ in range(iters):
        mid = (lo + hi) / 2
        ok = kl(p_hat, mid) <= budget
        lo = np.where(ok, mid, lo); hi = np.where(ok, hi, mid)
    return lo

print(kl(np.array(0.5), np.array(0.5)), klucb_index(np.array([0.5]), np.array([10]), 100))
''','''
import numpy as np
import matplotlib.pyplot as plt
def kl(p, q, eps=1e-12):
    p = np.clip(p, eps, 1 - eps); q = np.clip(q, eps, 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def klucb_index(p_hat, n, t, iters=25):
    lo, hi = p_hat.copy(), np.ones_like(p_hat)
    budget = np.log(max(t, 2)) / np.maximum(n, 1)
    for _ in range(iters):
        mid = (lo + hi) / 2
        ok = kl(p_hat, mid) <= budget
        lo = np.where(ok, mid, lo); hi = np.where(ok, hi, mid)
    return lo

print(kl(np.array(0.5), np.array(0.5)), klucb_index(np.array([0.5]), np.array([10]), 100))
''','''
assert abs(kl(np.array(0.5), np.array(0.5))) < 1e-12 and abs(kl(np.array(0.9), np.array(0.5)) - (0.9 * np.log(1.8) + 0.1 * np.log(0.2))) < 1e-9
idx = klucb_index(np.array([0.5]), np.array([10]), 100)[0]
assert 0.5 < idx < 1 and abs(10 * kl(np.array(0.5), np.array(idx)) - np.log(100)) < 1e-3
''', "p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))"),
S('Race UCB1, KL-UCB and Thompson',
"Thompson sampling: each arm has a Beta(successes + 1, failures + 1) posterior; sample from each and play the largest.",
'''
rng = np.random.default_rng(0)
runs, K, T = 200, 10, 3000
mu = rng.uniform(0.1, 0.9, (runs, K)); gaps = mu.max(axis=1, keepdims=True) - mu; R = np.arange(runs)

def run(select):
    N, Ssum, regret = np.zeros((runs, K)), np.zeros((runs, K)), np.zeros(T)
    for t in range(1, T + 1):
        a = select(N, Ssum, t)
        r = (rng.random(runs) < mu[R, a]).astype(float)
        N[R, a] += 1; Ssum[R, a] += r
        regret[t - 1] = gaps[R, a].mean()
    return np.cumsum(regret)

ucb1 = lambda N, Ssum, t: np.where(N == 0, np.inf, Ssum / np.maximum(N, 1) + np.sqrt(2 * np.log(t) / np.maximum(N, 1))).argmax(axis=1)
klucb = lambda N, Ssum, t: np.where(N == 0, np.inf, klucb_index(Ssum / np.maximum(N, 1), N, t)).argmax(axis=1)
thompson = lambda N, Ssum, t: rng.beta(___, ___).argmax(axis=1)

curves = {"UCB1": run(ucb1), "KL-UCB": run(klucb), "Thompson": run(thompson)}
for k, v in curves.items():
    plt.plot(v, label=k)
plt.xlabel("t"); plt.ylabel("cumulative regret"); plt.legend()
print({k: round(v[-1], 1) for k, v in curves.items()})
''','''
rng = np.random.default_rng(0)
runs, K, T = 200, 10, 3000
mu = rng.uniform(0.1, 0.9, (runs, K)); gaps = mu.max(axis=1, keepdims=True) - mu; R = np.arange(runs)

def run(select):
    N, Ssum, regret = np.zeros((runs, K)), np.zeros((runs, K)), np.zeros(T)
    for t in range(1, T + 1):
        a = select(N, Ssum, t)
        r = (rng.random(runs) < mu[R, a]).astype(float)
        N[R, a] += 1; Ssum[R, a] += r
        regret[t - 1] = gaps[R, a].mean()
    return np.cumsum(regret)

ucb1 = lambda N, Ssum, t: np.where(N == 0, np.inf, Ssum / np.maximum(N, 1) + np.sqrt(2 * np.log(t) / np.maximum(N, 1))).argmax(axis=1)
klucb = lambda N, Ssum, t: np.where(N == 0, np.inf, klucb_index(Ssum / np.maximum(N, 1), N, t)).argmax(axis=1)
thompson = lambda N, Ssum, t: rng.beta(Ssum + 1, N - Ssum + 1).argmax(axis=1)

curves = {"UCB1": run(ucb1), "KL-UCB": run(klucb), "Thompson": run(thompson)}
for k, v in curves.items():
    plt.plot(v, label=k)
plt.xlabel("t"); plt.ylabel("cumulative regret"); plt.legend()
print({k: round(v[-1], 1) for k, v in curves.items()})
''','''
assert curves["Thompson"][-1] < curves["UCB1"][-1] and curves["KL-UCB"][-1] < curves["UCB1"][-1], "Both should beat UCB1 on Bernoulli arms"
''', "Successes + 1 and failures + 1: rng.beta(Ssum + 1, N - Ssum + 1)"),
]),

dict(id='BA9', title='LinUCB: choosing bank offers from customer features', packages=['numpy', 'matplotlib'], intro=
"Contextual bandits: before choosing, you see who the customer is. LinUCB keeps a ridge-regression model per offer and adds an uncertainty bonus. This is the algorithm behind many real recommendation systems, including the famous Yahoo! news experiment.",
steps=[
S('Simulate customers and offers',
"Each customer is a 5-number feature vector (think: age band, balance, salary credit, card usage, digital activity, all scaled). Offer a's expected reward is x . theta_a, plus noise.",
'''
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
d, K, T = 5, 5, 3000
theta = rng.normal(0, 1, (K, d)) / np.sqrt(d)
X = rng.normal(0, 1, (T, d))
expected = X @ theta.T                      # T x K table of true expected rewards
best_reward = ___                           # reward of the best offer for every customer
print(expected.shape, best_reward[:3])
''','''
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
d, K, T = 5, 5, 3000
theta = rng.normal(0, 1, (K, d)) / np.sqrt(d)
X = rng.normal(0, 1, (T, d))
expected = X @ theta.T                      # T x K table of true expected rewards
best_reward = expected.max(axis=1)          # reward of the best offer for every customer
print(expected.shape, best_reward[:3])
''','''
assert np.allclose(best_reward, expected.max(axis=1))
''', "expected.max(axis=1)"),
S('LinUCB',
"Per offer: A = I + sum x x^T and b = sum r x. Estimate theta_hat = A^-1 b; score = theta_hat . x + alpha sqrt(x^T A^-1 x). Play the top score, then update that offer's A and b.",
'''
def linucb(alpha=1.0, seed=0):
    r_ = np.random.default_rng(seed)
    A = np.stack([np.eye(d)] * K); b = np.zeros((K, d)); regret = []
    for t in range(T):
        x = X[t]
        A_inv = np.linalg.inv(A)
        theta_hat = np.einsum("kij,kj->ki", A_inv, b)
        bonus = np.sqrt(np.einsum("i,kij,j->k", x, A_inv, x))
        a = int(np.argmax(theta_hat @ x + alpha * bonus))
        r = expected[t, a] + r_.normal(0, 0.5)
        A[a] += ___
        b[a] += r * x
        regret.append(best_reward[t] - expected[t, a])
    return np.cumsum(regret)

reg_lin = linucb()
print(reg_lin[-1])
''','''
def linucb(alpha=1.0, seed=0):
    r_ = np.random.default_rng(seed)
    A = np.stack([np.eye(d)] * K); b = np.zeros((K, d)); regret = []
    for t in range(T):
        x = X[t]
        A_inv = np.linalg.inv(A)
        theta_hat = np.einsum("kij,kj->ki", A_inv, b)
        bonus = np.sqrt(np.einsum("i,kij,j->k", x, A_inv, x))
        a = int(np.argmax(theta_hat @ x + alpha * bonus))
        r = expected[t, a] + r_.normal(0, 0.5)
        A[a] += np.outer(x, x)
        b[a] += r * x
        regret.append(best_reward[t] - expected[t, a])
    return np.cumsum(regret)

reg_lin = linucb()
print(reg_lin[-1])
''','''
assert reg_lin[-1] < 0.15 * best_reward.clip(0).sum() + 400, "LinUCB's regret should be modest"
assert reg_lin[-1] - reg_lin[-501] < reg_lin[500] - reg_lin[0], "Regret should grow more slowly later on"
''', "The outer product: np.outer(x, x)"),
S('Compare with random and context-blind epsilon-greedy',
"Context-blind epsilon-greedy learns one average reward per offer and ignores who the customer is. Random just picks offers uniformly.",
'''
def random_policy(seed=0):
    r_ = np.random.default_rng(seed)
    a = r_.integers(0, K, T)
    return np.cumsum(best_reward - expected[np.arange(T), a])

def blind_eps(eps=0.1, seed=0):
    r_ = np.random.default_rng(seed)
    N, S_ = np.zeros(K), np.zeros(K); regret = []
    for t in range(T):
        a = r_.integers(K) if r_.random() < eps else int(np.argmax(S_ / np.maximum(N, 1)))
        r = expected[t, a] + r_.normal(0, 0.5)
        N[a] += 1; S_[a] += r
        regret.append(best_reward[t] - expected[t, a])
    return np.cumsum(regret)

curves = {"LinUCB": reg_lin, "context-blind eps-greedy": blind_eps(), "random": ___}
for k, v in curves.items():
    plt.plot(v, label=k)
plt.xlabel("customers"); plt.ylabel("cumulative regret"); plt.legend()
print({k: round(v[-1]) for k, v in curves.items()})
''','''
def random_policy(seed=0):
    r_ = np.random.default_rng(seed)
    a = r_.integers(0, K, T)
    return np.cumsum(best_reward - expected[np.arange(T), a])

def blind_eps(eps=0.1, seed=0):
    r_ = np.random.default_rng(seed)
    N, S_ = np.zeros(K), np.zeros(K); regret = []
    for t in range(T):
        a = r_.integers(K) if r_.random() < eps else int(np.argmax(S_ / np.maximum(N, 1)))
        r = expected[t, a] + r_.normal(0, 0.5)
        N[a] += 1; S_[a] += r
        regret.append(best_reward[t] - expected[t, a])
    return np.cumsum(regret)

curves = {"LinUCB": reg_lin, "context-blind eps-greedy": blind_eps(), "random": random_policy()}
for k, v in curves.items():
    plt.plot(v, label=k)
plt.xlabel("customers"); plt.ylabel("cumulative regret"); plt.legend()
print({k: round(v[-1]) for k, v in curves.items()})
''','''
assert curves["LinUCB"][-1] < 0.5 * curves["context-blind eps-greedy"][-1] < curves["random"][-1], "Using context should cut regret dramatically"
''', "random_policy()"),
]),

dict(id='BA10', title='Watching a confidence ellipsoid shrink', packages=['numpy', 'matplotlib'], intro=
"In a linear bandit, everything LinUCB knows about theta is an ellipse: the set of parameters consistent with the data so far. Draw it at t = 10, 100 and 1000 and confirm the truth stays inside.",
steps=[
S('The ellipse',
"After t observations, V = lambda I + sum x x^T and theta_hat = V^-1 sum r x. The confidence set is {theta : (theta - theta_hat)^T V (theta - theta_hat) <= beta}. Its area is proportional to 1 / sqrt(det V).",
'''
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
theta_star = np.array([0.6, -0.4]); sigma, lam = 0.3, 1.0
Xs = rng.normal(size=(1000, 2))
rs = Xs @ theta_star + rng.normal(0, sigma, 1000)

def ellipse(t, delta=0.05):
    X, r = Xs[:t], rs[:t]
    V = lam * np.eye(2) + X.T @ X
    theta_hat = np.linalg.solve(V, X.T @ r)
    beta = (np.sqrt(lam) * 1.0 + sigma * np.sqrt(2 * np.log(1 / delta) + 2 * np.log(1 + t / (2 * lam)))) ** 2
    inside = ___ <= beta
    return V, theta_hat, beta, inside

for t in [10, 100, 1000]:
    V, th, beta, inside = ellipse(t)
    print(t, th.round(3), inside, 1 / np.sqrt(np.linalg.det(V)))
''','''
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
theta_star = np.array([0.6, -0.4]); sigma, lam = 0.3, 1.0
Xs = rng.normal(size=(1000, 2))
rs = Xs @ theta_star + rng.normal(0, sigma, 1000)

def ellipse(t, delta=0.05):
    X, r = Xs[:t], rs[:t]
    V = lam * np.eye(2) + X.T @ X
    theta_hat = np.linalg.solve(V, X.T @ r)
    beta = (np.sqrt(lam) * 1.0 + sigma * np.sqrt(2 * np.log(1 / delta) + 2 * np.log(1 + t / (2 * lam)))) ** 2
    inside = (theta_star - theta_hat) @ V @ (theta_star - theta_hat) <= beta
    return V, theta_hat, beta, inside

for t in [10, 100, 1000]:
    V, th, beta, inside = ellipse(t)
    print(t, th.round(3), inside, 1 / np.sqrt(np.linalg.det(V)))
''','''
assert all(ellipse(t)[3] for t in [10, 100, 1000]), "The true theta should lie inside every confidence ellipse"
areas = [1 / np.sqrt(np.linalg.det(ellipse(t)[0])) for t in [10, 100, 1000]]
assert areas[0] > areas[1] > areas[2]
''', "The quadratic form: (theta_star - theta_hat) @ V @ (theta_star - theta_hat)"),
F(title='Draw them',
text="Each ellipse is drawn from V's eigenvectors. The star is the true theta.",
starter='''
ang = np.linspace(0, 2 * np.pi, 200)
for t in [10, 100, 1000]:
    V, th, beta, _ = ellipse(t)
    evals, evecs = np.linalg.eigh(V)
    pts = th[:, None] + evecs @ (np.sqrt(beta / evals)[:, None] * np.vstack([np.cos(ang), np.sin(ang)]))
    plt.plot(pts[0], pts[1], label=f"t = {t}")
plt.plot(*theta_star, "k*", ms=12); plt.axis("equal"); plt.legend()
''', solution='''
ang = np.linspace(0, 2 * np.pi, 200)
for t in [10, 100, 1000]:
    V, th, beta, _ = ellipse(t)
    evals, evecs = np.linalg.eigh(V)
    pts = th[:, None] + evecs @ (np.sqrt(beta / evals)[:, None] * np.vstack([np.cos(ang), np.sin(ang)]))
    plt.plot(pts[0], pts[1], label=f"t = {t}")
plt.plot(*theta_star, "k*", ms=12); plt.axis("equal"); plt.legend()
''', check="assert len(plt.gca().lines) == 4", hint="Nothing to fill in: run it."),
]),

dict(id='BA11', title='Exp4: learning from expert advice with bandit feedback', packages=['numpy'], intro=
"Exp4 doesn't learn a model of rewards at all. It is given a handful of 'experts' (fixed policies that recommend an offer for each customer) and learns which expert to trust, using importance weighting like Exp3.",
steps=[
S('Exp4',
"Each round, every expert gives a recommendation; mix experts by weight into a probability over offers; play one; estimate each offer's reward by importance weighting; credit each expert with the estimated reward of what it recommended.",
'''
import numpy as np
rng = np.random.default_rng(0)
d, K, T = 5, 5, 4000
theta = rng.normal(0, 1, (K, d)) / np.sqrt(d)
X = rng.normal(0, 1, (T, d))
probs = 1 / (1 + np.exp(-3 * (X @ theta.T)))                 # reward probabilities, T x K
experts = [
    lambda x: int(np.argmax(x @ theta.T + rng.normal(0, 0.3, K))),   # a decent model
    lambda x: int(np.argmax(x[:K])),                               # a crude heuristic
    lambda x: 0,                                                   # always offer 0
    lambda x: int(rng.integers(K)),                                # random
]
E = len(experts)
eta = np.sqrt(2 * np.log(E) / (T * K))
logw, got = np.zeros(E), 0.0
expert_reward = np.zeros(E)
for t in range(T):
    advice = np.array([e(X[t]) for e in experts])
    q = np.exp(logw - logw.max()); q /= q.sum()
    p = np.bincount(advice, weights=q, minlength=K)
    a = rng.choice(K, p=p)
    r = float(rng.random() < probs[t, a])
    got += r
    expert_reward += probs[t, advice]
    r_hat = np.zeros(K); r_hat[a] = r / p[a]
    logw += eta * ___                     # each expert earns the estimated reward of its advice
print(got, expert_reward.round(), np.exp(logw - logw.max()).round(3))
''','''
import numpy as np
rng = np.random.default_rng(0)
d, K, T = 5, 5, 4000
theta = rng.normal(0, 1, (K, d)) / np.sqrt(d)
X = rng.normal(0, 1, (T, d))
probs = 1 / (1 + np.exp(-3 * (X @ theta.T)))                 # reward probabilities, T x K
experts = [
    lambda x: int(np.argmax(x @ theta.T + rng.normal(0, 0.3, K))),   # a decent model
    lambda x: int(np.argmax(x[:K])),                               # a crude heuristic
    lambda x: 0,                                                   # always offer 0
    lambda x: int(rng.integers(K)),                                # random
]
E = len(experts)
eta = np.sqrt(2 * np.log(E) / (T * K))
logw, got = np.zeros(E), 0.0
expert_reward = np.zeros(E)
for t in range(T):
    advice = np.array([e(X[t]) for e in experts])
    q = np.exp(logw - logw.max()); q /= q.sum()
    p = np.bincount(advice, weights=q, minlength=K)
    a = rng.choice(K, p=p)
    r = float(rng.random() < probs[t, a])
    got += r
    expert_reward += probs[t, advice]
    r_hat = np.zeros(K); r_hat[a] = r / p[a]
    logw += eta * r_hat[advice]                     # each expert earns the estimated reward of its advice
print(got, expert_reward.round(), np.exp(logw - logw.max()).round(3))
''','''
assert np.argmax(logw) == 0, "Exp4 should come to trust the good model (expert 0) most"
assert got > expert_reward[3], "Exp4 should beat the random expert"
''', "Index the estimates by each expert's advice: r_hat[advice]"),
]),

dict(id='BA12', title='Best-arm identification: uniform vs LUCB', packages=['numpy', 'matplotlib'], intro=
"Sometimes you don't care about reward along the way: you want to find the best arm with confidence 95% using as few samples as possible (an A/B/n test). LUCB focuses its samples on the two arms that are hardest to tell apart.",
steps=[
S('Hoeffding confidence bounds',
"With n samples of a [0, 1] reward, the true mean is within sqrt(ln(4 K n^2 / delta) / (2 n)) of the sample mean for all arms and times, with probability at least 1 - delta. Write the radius.",
'''
import numpy as np
import matplotlib.pyplot as plt
def radius(n, K, delta):
    return ___

print(radius(np.array([10, 100, 1000]), 5, 0.05))
''','''
import numpy as np
import matplotlib.pyplot as plt
def radius(n, K, delta):
    return np.sqrt(np.log(4 * K * n ** 2 / delta) / (2 * n))

print(radius(np.array([10, 100, 1000]), 5, 0.05))
''','''
r = radius(np.array([10, 100, 1000]), 5, 0.05)
assert np.allclose(r, np.sqrt(np.log(4 * 5 * np.array([10, 100, 1000]) ** 2 / 0.05) / (2 * np.array([10, 100, 1000])))) and r[0] > r[1] > r[2]
''', "np.sqrt(np.log(4 * K * n ** 2 / delta) / (2 * n))"),
S('Uniform sampling vs LUCB',
"Uniform: pull every arm in turn until one arm's lower bound beats every other arm's upper bound. LUCB: each round, sample only the current leader and its strongest challenger (highest upper bound among the rest).",
'''
def identify(mu, strategy, delta=0.05, seed=0, max_pulls=200000):
    rng = np.random.default_rng(seed)
    K = len(mu); N = np.zeros(K); Ssum = np.zeros(K)
    for a in range(K):
        N[a] += 1; Ssum[a] += rng.random() < mu[a]
    while N.sum() < max_pulls:
        m = Ssum / N; rad = radius(N, K, delta)
        lead = int(np.argmax(m))
        others = [a for a in range(K) if a != lead]
        challenger = others[int(np.argmax((m + rad)[others]))]
        if m[lead] - rad[lead] > m[challenger] + rad[challenger]:
            return lead, int(N.sum())
        to_pull = range(K) if strategy == "uniform" else ___
        for a in to_pull:
            N[a] += 1; Ssum[a] += rng.random() < mu[a]
    return lead, int(N.sum())

mu = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
print(identify(mu, "uniform"), identify(mu, "lucb"))
''','''
def identify(mu, strategy, delta=0.05, seed=0, max_pulls=200000):
    rng = np.random.default_rng(seed)
    K = len(mu); N = np.zeros(K); Ssum = np.zeros(K)
    for a in range(K):
        N[a] += 1; Ssum[a] += rng.random() < mu[a]
    while N.sum() < max_pulls:
        m = Ssum / N; rad = radius(N, K, delta)
        lead = int(np.argmax(m))
        others = [a for a in range(K) if a != lead]
        challenger = others[int(np.argmax((m + rad)[others]))]
        if m[lead] - rad[lead] > m[challenger] + rad[challenger]:
            return lead, int(N.sum())
        to_pull = range(K) if strategy == "uniform" else [lead, challenger]
        for a in to_pull:
            N[a] += 1; Ssum[a] += rng.random() < mu[a]
    return lead, int(N.sum())

mu = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
print(identify(mu, "uniform"), identify(mu, "lucb"))
''','''
(u_arm, u_n), (l_arm, l_n) = identify(mu, "uniform"), identify(mu, "lucb")
assert u_arm == 0 and l_arm == 0, "Both should find arm 0"
assert l_n < u_n, "LUCB should need fewer samples than uniform sampling"
''', "Just the leader and the challenger: [lead, challenger]"),
S('Samples needed vs the gap',
"Shrink the gap between the best and second-best arm and count samples. Theory says the cost grows like 1 / gap^2.",
'''
gaps = [0.3, 0.2, 0.1]
counts = []
for g in gaps:
    mu_g = np.array([0.5, 0.5 - g, 0.2, 0.1])
    counts.append(identify(mu_g, "lucb")[1])
plt.loglog(gaps, counts, "o-"); plt.xlabel("gap"); plt.ylabel("samples needed")
ratio = ___     # samples at gap 0.1 divided by samples at gap 0.2
print(counts, ratio)
''','''
gaps = [0.3, 0.2, 0.1]
counts = []
for g in gaps:
    mu_g = np.array([0.5, 0.5 - g, 0.2, 0.1])
    counts.append(identify(mu_g, "lucb")[1])
plt.loglog(gaps, counts, "o-"); plt.xlabel("gap"); plt.ylabel("samples needed")
ratio = counts[2] / counts[1]     # samples at gap 0.1 divided by samples at gap 0.2
print(counts, ratio)
''','''
assert counts[0] < counts[1] < counts[2] and 2.5 < ratio < 8, "Halving the gap should multiply the samples by roughly 4"
''', "counts[2] / counts[1]"),
]),
]

P16 = dict(id='P16', title='Contextual bandit for bank offers, with offline evaluation', packages=['numpy', 'matplotlib'], intro=
"A realistic project: simulated customers, five product offers with click-through probabilities that depend on the customer, three online policies, and then the question every bank's model-risk team will ask: how do you evaluate a new policy using only logs from the old one? You'll use the replay method (Li et al., 2011).",
steps=[
S('Simulate customers and offers',
"Features: intercept, age (scaled), balance (scaled), salaried (0/1), digital user (0/1). Each offer's acceptance probability is a logistic function of the features. Fill in the logistic.",
'''
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
offers = ["credit card", "personal loan", "fixed deposit", "insurance", "mutual fund SIP"]
K, d = len(offers), 5
W = np.array([[-2.0, -0.5, 0.2, 1.0, 1.2],
              [-2.2, 0.3, -0.8, 1.2, 0.2],
              [-2.0, 1.0, 1.0, 0.0, -0.4],
              [-2.4, 0.8, 0.3, 0.5, 0.0],
              [-2.1, -0.6, 0.6, 0.6, 1.0]])

def customers(n, rng):
    return np.column_stack([np.ones(n), rng.normal(0, 1, n), rng.normal(0, 1, n), rng.random(n) < 0.6, rng.random(n) < 0.5]).astype(float)

def accept_prob(X):
    return ___        # n x K acceptance probabilities

X = customers(5, rng)
print(accept_prob(X).round(3))
''','''
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)
offers = ["credit card", "personal loan", "fixed deposit", "insurance", "mutual fund SIP"]
K, d = len(offers), 5
W = np.array([[-2.0, -0.5, 0.2, 1.0, 1.2],
              [-2.2, 0.3, -0.8, 1.2, 0.2],
              [-2.0, 1.0, 1.0, 0.0, -0.4],
              [-2.4, 0.8, 0.3, 0.5, 0.0],
              [-2.1, -0.6, 0.6, 0.6, 1.0]])

def customers(n, rng):
    return np.column_stack([np.ones(n), rng.normal(0, 1, n), rng.normal(0, 1, n), rng.random(n) < 0.6, rng.random(n) < 0.5]).astype(float)

def accept_prob(X):
    return 1 / (1 + np.exp(-(X @ W.T)))        # n x K acceptance probabilities

X = customers(5, rng)
print(accept_prob(X).round(3))
''','''
P = accept_prob(X)
assert P.shape == (5, 5) and np.allclose(P, 1 / (1 + np.exp(-(X @ W.T))))
''', "The sigmoid of X @ W.T"),
S('Three online policies',
"LinUCB (from BA9), linear Thompson sampling (sample theta from N(theta_hat, v^2 A^-1) per offer), and epsilon-greedy over per-offer ridge regressions. Fill in the Thompson sample.",
'''
T = 4000
Xon = customers(T, np.random.default_rng(1))
Pon = accept_prob(Xon)

def run(policy, seed=0):
    r_ = np.random.default_rng(seed)
    A = np.stack([np.eye(d)] * K); b = np.zeros((K, d)); accepted = []
    for t in range(T):
        x = Xon[t]
        A_inv = np.linalg.inv(A); th = np.einsum("kij,kj->ki", A_inv, b)
        if policy == "linucb":
            score = th @ x + 0.5 * np.sqrt(np.einsum("i,kij,j->k", x, A_inv, x))
        elif policy == "thompson":
            samples = np.array([r_.multivariate_normal(th[k], ___) for k in range(K)])
            score = samples @ x
        else:
            score = th @ x if r_.random() > 0.1 else r_.random(K)
        a = int(np.argmax(score))
        r = float(r_.random() < Pon[t, a])
        A[a] += np.outer(x, x); b[a] += r * x
        accepted.append(r)
    return np.array(accepted)

results = {p: run(p) for p in ["linucb", "thompson", "eps-greedy"]}
oracle = Pon.max(axis=1).mean(); random_rate = Pon.mean()
for p, acc in results.items():
    plt.plot(np.cumsum(acc) / np.arange(1, T + 1), label=p)
plt.axhline(oracle, ls="--", c="k", label="oracle"); plt.axhline(random_rate, ls=":", c="gray", label="random")
plt.xlabel("customers"); plt.ylabel("acceptance rate so far"); plt.legend()
print({p: round(a.mean(), 3) for p, a in results.items()}, round(oracle, 3), round(random_rate, 3))
''','''
T = 4000
Xon = customers(T, np.random.default_rng(1))
Pon = accept_prob(Xon)

def run(policy, seed=0):
    r_ = np.random.default_rng(seed)
    A = np.stack([np.eye(d)] * K); b = np.zeros((K, d)); accepted = []
    for t in range(T):
        x = Xon[t]
        A_inv = np.linalg.inv(A); th = np.einsum("kij,kj->ki", A_inv, b)
        if policy == "linucb":
            score = th @ x + 0.5 * np.sqrt(np.einsum("i,kij,j->k", x, A_inv, x))
        elif policy == "thompson":
            samples = np.array([r_.multivariate_normal(th[k], 0.25 * A_inv[k]) for k in range(K)])
            score = samples @ x
        else:
            score = th @ x if r_.random() > 0.1 else r_.random(K)
        a = int(np.argmax(score))
        r = float(r_.random() < Pon[t, a])
        A[a] += np.outer(x, x); b[a] += r * x
        accepted.append(r)
    return np.array(accepted)

results = {p: run(p) for p in ["linucb", "thompson", "eps-greedy"]}
oracle = Pon.max(axis=1).mean(); random_rate = Pon.mean()
for p, acc in results.items():
    plt.plot(np.cumsum(acc) / np.arange(1, T + 1), label=p)
plt.axhline(oracle, ls="--", c="k", label="oracle"); plt.axhline(random_rate, ls=":", c="gray", label="random")
plt.xlabel("customers"); plt.ylabel("acceptance rate so far"); plt.legend()
print({p: round(a.mean(), 3) for p, a in results.items()}, round(oracle, 3), round(random_rate, 3))
''','''
assert all(a.mean() > random_rate + 0.02 for a in results.values()), "Every learning policy should beat random offers"
''', "Covariance v^2 A^-1 with v = 0.5: 0.25 * A_inv[k]"),
S('Offline evaluation with the replay method',
"Pretend the bank logged 20,000 customers under a uniformly random offer policy. To evaluate a new policy offline, walk through the log, ask the policy what it would offer, and keep only the events where it matches the logged offer. Because logging was uniform, the matched events are an unbiased sample.",
'''
N_log = 20000
Xlog = customers(N_log, np.random.default_rng(7))
logged_offer = np.random.default_rng(8).integers(0, K, N_log)
logged_reward = (np.random.default_rng(9).random(N_log) < accept_prob(Xlog)[np.arange(N_log), logged_offer]).astype(float)

def replay_linucb(alpha=0.5):
    A = np.stack([np.eye(d)] * K); b = np.zeros((K, d)); kept = []
    for t in range(N_log):
        x = Xlog[t]
        A_inv = np.linalg.inv(A); th = np.einsum("kij,kj->ki", A_inv, b)
        a = int(np.argmax(th @ x + alpha * np.sqrt(np.einsum("i,kij,j->k", x, A_inv, x))))
        if ___:
            r = logged_reward[t]
            kept.append(r)
            A[a] += np.outer(x, x); b[a] += r * x
    return np.mean(kept), len(kept)

offline_estimate, n_used = replay_linucb()
online_truth = results["linucb"][:n_used].mean()
print(round(offline_estimate, 3), n_used, round(online_truth, 3))
''','''
N_log = 20000
Xlog = customers(N_log, np.random.default_rng(7))
logged_offer = np.random.default_rng(8).integers(0, K, N_log)
logged_reward = (np.random.default_rng(9).random(N_log) < accept_prob(Xlog)[np.arange(N_log), logged_offer]).astype(float)

def replay_linucb(alpha=0.5):
    A = np.stack([np.eye(d)] * K); b = np.zeros((K, d)); kept = []
    for t in range(N_log):
        x = Xlog[t]
        A_inv = np.linalg.inv(A); th = np.einsum("kij,kj->ki", A_inv, b)
        a = int(np.argmax(th @ x + alpha * np.sqrt(np.einsum("i,kij,j->k", x, A_inv, x))))
        if a == logged_offer[t]:
            r = logged_reward[t]
            kept.append(r)
            A[a] += np.outer(x, x); b[a] += r * x
    return np.mean(kept), len(kept)

offline_estimate, n_used = replay_linucb()
online_truth = results["linucb"][:n_used].mean()
print(round(offline_estimate, 3), n_used, round(online_truth, 3))
''','''
assert 0.15 * N_log < n_used < 0.25 * N_log, "About 1 in K logged events should match"
assert abs(offline_estimate - online_truth) < 0.04, "The replay estimate should be close to the online result"
''', "Keep the event only when the policy agrees with the log: a == logged_offer[t]"),
S('Write it up',
"In two or three sentences for a manager: which policy you'd pilot, its acceptance rate vs random offers, and how the offline replay supports the decision. Use an f-string so the numbers fill themselves in.",
'''
summary = f"""
___
"""
print(summary)
''','''
best = max(results, key=lambda p: results[p].mean())
summary = f"""
Across {T} simulated customers, {best} reached a {results[best].mean():.1%} acceptance rate versus {random_rate:.1%} for random offers
(an oracle that knew every customer's preferences would get {oracle:.1%}). Replaying LinUCB on {N_log} logged random-offer events
estimates {offline_estimate:.1%}, close to its online result, so we can validate new policies on existing logs before a live pilot.
"""
print(summary)
''','''
assert "___" not in summary and len(summary.split()) >= 30, "Write two or three full sentences"
assert "%" in summary, "Include the acceptance rates as percentages"
''', "Pick the best policy with max(results, key=lambda p: results[p].mean()) and format rates with {x:.1%}"),
])

BA_ALL = BA + [P16]
