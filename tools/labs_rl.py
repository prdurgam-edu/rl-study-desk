"""Guided browser labs for the Reinforcement Learning projects P09-P12, P14, P15."""
from labs_warmup import S
from labs_dl import F

RL = [
dict(id='P09', title='The 10-armed bandit testbed', packages=['numpy', 'matplotlib'], intro=
"Reproduce Sutton & Barto's Figures 2.2-2.4: epsilon-greedy, optimistic initial values and UCB on the 10-armed testbed, then add Thompson sampling. Everything is vectorised over 500 independent runs so it's fast.",
steps=[
S('Epsilon-greedy with incremental averages',
"Each run has 10 arms with true values q*(a) ~ N(0, 1). Keep an estimate Q(a) and count N(a). After pulling a and getting reward r: Q(a) += (r - Q(a)) / N(a). That's a running mean, like AVG() updated one row at a time.",
'''
import numpy as np
import matplotlib.pyplot as plt
runs, k, steps = 500, 10, 1000
rng = np.random.default_rng(0)
q_true = rng.normal(size=(runs, k))
best = q_true.argmax(axis=1)
R = np.arange(runs)

def run(select, Q0=0.0):
    Q = np.full((runs, k), Q0); N = np.zeros((runs, k))
    rewards, optimal = np.zeros(steps), np.zeros(steps)
    for t in range(1, steps + 1):
        a = select(Q, N, t)
        r = q_true[R, a] + rng.normal(size=runs)
        N[R, a] += 1
        Q[R, a] += ___
        rewards[t - 1] = r.mean(); optimal[t - 1] = (a == best).mean()
    return rewards, optimal

def eps_greedy(eps):
    def select(Q, N, t):
        explore = rng.random(runs) < eps
        return np.where(explore, rng.integers(0, k, runs), Q.argmax(axis=1))
    return select

res = {f"eps={e}": run(eps_greedy(e)) for e in [0.0, 0.01, 0.1]}
for name, (rw, op) in res.items():
    plt.plot(rw, label=name)
plt.xlabel("step"); plt.ylabel("average reward"); plt.legend()
''','''
import numpy as np
import matplotlib.pyplot as plt
runs, k, steps = 500, 10, 1000
rng = np.random.default_rng(0)
q_true = rng.normal(size=(runs, k))
best = q_true.argmax(axis=1)
R = np.arange(runs)

def run(select, Q0=0.0):
    Q = np.full((runs, k), Q0); N = np.zeros((runs, k))
    rewards, optimal = np.zeros(steps), np.zeros(steps)
    for t in range(1, steps + 1):
        a = select(Q, N, t)
        r = q_true[R, a] + rng.normal(size=runs)
        N[R, a] += 1
        Q[R, a] += (r - Q[R, a]) / N[R, a]
        rewards[t - 1] = r.mean(); optimal[t - 1] = (a == best).mean()
    return rewards, optimal

def eps_greedy(eps):
    def select(Q, N, t):
        explore = rng.random(runs) < eps
        return np.where(explore, rng.integers(0, k, runs), Q.argmax(axis=1))
    return select

res = {f"eps={e}": run(eps_greedy(e)) for e in [0.0, 0.01, 0.1]}
for name, (rw, op) in res.items():
    plt.plot(rw, label=name)
plt.xlabel("step"); plt.ylabel("average reward"); plt.legend()
''','''
late = {n: v[0][-300:].mean() for n, v in res.items()}
assert late["eps=0.1"] > late["eps=0.0"], "Exploring (eps=0.1) should beat pure greedy in the long run, as in Figure 2.2"
''', "(r - Q[R, a]) / N[R, a]"),
S('Optimistic initial values',
"Start every estimate at +5 and act greedily. Every arm disappoints at first, so the agent tries them all: exploration without randomness (Figure 2.3).",
'''
opt_rw, opt_op = run(eps_greedy(0.0), Q0=___)
plt.plot(opt_op, label="optimistic, greedy"); plt.plot(res["eps=0.1"][1], label="realistic, eps=0.1")
plt.xlabel("step"); plt.ylabel("% optimal action"); plt.legend()
print(opt_op[-200:].mean(), res["eps=0.1"][1][-200:].mean())
''','''
opt_rw, opt_op = run(eps_greedy(0.0), Q0=5.0)
plt.plot(opt_op, label="optimistic, greedy"); plt.plot(res["eps=0.1"][1], label="realistic, eps=0.1")
plt.xlabel("step"); plt.ylabel("% optimal action"); plt.legend()
print(opt_op[-200:].mean(), res["eps=0.1"][1][-200:].mean())
''','''
assert opt_op[-200:].mean() > res["eps=0.0"][1][-200:].mean() + 0.2, "Optimism should fix greedy's exploration problem"
''', "Q0=5.0"),
S('Upper confidence bound',
"UCB picks argmax Q(a) + c sqrt(ln t / N(a)): arms tried rarely get a bonus. Untried arms (N = 0) go first.",
'''
def ucb(c):
    def select(Q, N, t):
        bonus = c * np.sqrt(np.log(t) / np.maximum(N, 1e-9))
        score = np.where(N == 0, np.inf, ___)
        return score.argmax(axis=1)
    return select

ucb_rw, ucb_op = run(ucb(2.0))
plt.plot(ucb_rw, label="UCB c=2"); plt.plot(res["eps=0.1"][0], label="eps=0.1")
plt.xlabel("step"); plt.ylabel("average reward"); plt.legend()
print(ucb_rw[-500:].mean(), res["eps=0.1"][0][-500:].mean())
''','''
def ucb(c):
    def select(Q, N, t):
        bonus = c * np.sqrt(np.log(t) / np.maximum(N, 1e-9))
        score = np.where(N == 0, np.inf, Q + bonus)
        return score.argmax(axis=1)
    return select

ucb_rw, ucb_op = run(ucb(2.0))
plt.plot(ucb_rw, label="UCB c=2"); plt.plot(res["eps=0.1"][0], label="eps=0.1")
plt.xlabel("step"); plt.ylabel("average reward"); plt.legend()
print(ucb_rw[-500:].mean(), res["eps=0.1"][0][-500:].mean())
''','''
assert ucb_rw[-500:].mean() > res["eps=0.1"][0][-500:].mean(), "UCB should beat eps-greedy (Figure 2.4)"
''', "Q + bonus"),
S('Thompson sampling',
"Keep a Gaussian belief about each arm: mean Q(a), variance 1 / (N(a) + 1). Each step, sample one value from every belief and play the largest sample.",
'''
def thompson(Q, N, t):
    samples = Q + rng.normal(size=Q.shape) * ___
    return samples.argmax(axis=1)

ts_rw, ts_op = run(thompson)
for name, curve in [("eps=0.1", res["eps=0.1"][1]), ("UCB", ucb_op), ("Thompson", ts_op)]:
    plt.plot(curve, label=name)
plt.xlabel("step"); plt.ylabel("% optimal action"); plt.legend()
print(ts_op[-200:].mean())
''','''
def thompson(Q, N, t):
    samples = Q + rng.normal(size=Q.shape) * np.sqrt(1 / (N + 1))
    return samples.argmax(axis=1)

ts_rw, ts_op = run(thompson)
for name, curve in [("eps=0.1", res["eps=0.1"][1]), ("UCB", ucb_op), ("Thompson", ts_op)]:
    plt.plot(curve, label=name)
plt.xlabel("step"); plt.ylabel("% optimal action"); plt.legend()
print(ts_op[-200:].mean())
''','''
assert ts_op[-200:].mean() > res["eps=0.1"][1][-200:].mean(), "Thompson sampling should pick the best arm more often than eps-greedy"
''', "The standard deviation is the square root of the variance: np.sqrt(1 / (N + 1))"),
]),

dict(id='P10', title='Grid-world dynamic programming', packages=['numpy', 'matplotlib'], intro=
"Sutton & Barto's Example 4.1: a 4x4 grid, terminal states in two corners, -1 reward per step. Evaluate the random policy, then find the optimal one by policy iteration and value iteration.",
steps=[
S('The environment',
"States 0..15 row by row; 0 and 15 are terminal. Moving off the grid leaves you where you are. Fill in the move for 'down'.",
'''
import numpy as np
import matplotlib.pyplot as plt
N_S, ACTIONS, TERMINAL, gamma = 16, ["up", "right", "down", "left"], {0, 15}, 1.0

def next_state(s, a):
    r, c = divmod(s, 4)
    if a == "up":    r = max(r - 1, 0)
    if a == "right": c = min(c + 1, 3)
    if a == "down":  r = ___
    if a == "left":  c = max(c - 1, 0)
    return r * 4 + c

print([next_state(5, a) for a in ACTIONS], next_state(14, "down"))
''','''
import numpy as np
import matplotlib.pyplot as plt
N_S, ACTIONS, TERMINAL, gamma = 16, ["up", "right", "down", "left"], {0, 15}, 1.0

def next_state(s, a):
    r, c = divmod(s, 4)
    if a == "up":    r = max(r - 1, 0)
    if a == "right": c = min(c + 1, 3)
    if a == "down":  r = min(r + 1, 3)
    if a == "left":  c = max(c - 1, 0)
    return r * 4 + c

print([next_state(5, a) for a in ACTIONS], next_state(14, "down"))
''','''
assert [next_state(5, a) for a in ACTIONS] == [1, 6, 9, 4] and next_state(14, "down") == 14
''', "min(r + 1, 3)"),
S('Evaluate the random policy',
"Iterative policy evaluation: V(s) = average over the 4 actions of [-1 + V(next)]. Sweep until nothing changes by more than 1e-4. You should reproduce Figure 4.1: -14, -18, -20, -22.",
'''
def evaluate_random(theta=1e-4):
    V = np.zeros(N_S)
    while True:
        delta = 0
        for s in range(N_S):
            if s in TERMINAL:
                continue
            v_new = ___
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            return V

V_random = evaluate_random()
print(V_random.reshape(4, 4).round(0))
''','''
def evaluate_random(theta=1e-4):
    V = np.zeros(N_S)
    while True:
        delta = 0
        for s in range(N_S):
            if s in TERMINAL:
                continue
            v_new = np.mean([-1 + gamma * V[next_state(s, a)] for a in ACTIONS])
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            return V

V_random = evaluate_random()
print(V_random.reshape(4, 4).round(0))
''','''
expected = np.array([[0, -14, -20, -22], [-14, -18, -20, -20], [-20, -20, -18, -14], [-22, -20, -14, 0]])
assert np.array_equal(V_random.reshape(4, 4).round(0), expected), "Should match Sutton & Barto Figure 4.1"
''', "np.mean([-1 + gamma * V[next_state(s, a)] for a in ACTIONS])"),
S('Value iteration',
"Replace the average with a max: V(s) = max over actions of [-1 + V(next)]. The result is the optimal value: minus the number of steps to the nearest corner.",
'''
def value_iteration(theta=1e-6):
    V, sweeps = np.zeros(N_S), 0
    while True:
        sweeps += 1; delta = 0
        for s in range(N_S):
            if s in TERMINAL:
                continue
            v_new = ___
            delta = max(delta, abs(v_new - V[s])); V[s] = v_new
        if delta < theta:
            return V, sweeps

V_star, sweeps = value_iteration()
policy = {s: max(ACTIONS, key=lambda a: -1 + V_star[next_state(s, a)]) for s in range(N_S) if s not in TERMINAL}
print(V_star.reshape(4, 4), sweeps)
''','''
def value_iteration(theta=1e-6):
    V, sweeps = np.zeros(N_S), 0
    while True:
        sweeps += 1; delta = 0
        for s in range(N_S):
            if s in TERMINAL:
                continue
            v_new = max(-1 + gamma * V[next_state(s, a)] for a in ACTIONS)
            delta = max(delta, abs(v_new - V[s])); V[s] = v_new
        if delta < theta:
            return V, sweeps

V_star, sweeps = value_iteration()
policy = {s: max(ACTIONS, key=lambda a: -1 + V_star[next_state(s, a)]) for s in range(N_S) if s not in TERMINAL}
print(V_star.reshape(4, 4), sweeps)
''','''
steps_to_corner = np.array([min(r + c, (3 - r) + (3 - c)) for r in range(4) for c in range(4)])
assert np.array_equal(V_star, -steps_to_corner), "Optimal values are minus the distance to the nearest terminal"
''', "max(-1 + gamma * V[next_state(s, a)] for a in ACTIONS)"),
S('Policy iteration',
"Alternate: evaluate the current policy exactly, then make it greedy with respect to those values. Stop when the policy stops changing. Start from 'always go up'.",
'''
def evaluate(pi, theta=1e-8):
    V = np.zeros(N_S)
    for _ in range(1000):
        delta = 0
        for s in range(N_S):
            if s in TERMINAL: continue
            v_new = -1 + V[next_state(s, pi[s])]
            delta = max(delta, abs(v_new - V[s])); V[s] = v_new
        if delta < theta: break
    return V

pi = {s: "up" for s in range(N_S) if s not in TERMINAL}
for iteration in range(20):
    V_pi = evaluate(pi)
    new_pi = {s: max(ACTIONS, key=lambda a: -1 + V_pi[next_state(s, a)]) for s in pi}
    if ___:
        break
    pi = new_pi
print(iteration, evaluate(pi).reshape(4, 4))
''','''
def evaluate(pi, theta=1e-8):
    V = np.zeros(N_S)
    for _ in range(1000):
        delta = 0
        for s in range(N_S):
            if s in TERMINAL: continue
            v_new = -1 + V[next_state(s, pi[s])]
            delta = max(delta, abs(v_new - V[s])); V[s] = v_new
        if delta < theta: break
    return V

pi = {s: "up" for s in range(N_S) if s not in TERMINAL}
for iteration in range(20):
    V_pi = evaluate(pi)
    new_pi = {s: max(ACTIONS, key=lambda a: -1 + V_pi[next_state(s, a)]) for s in pi}
    if new_pi == pi:
        break
    pi = new_pi
print(iteration, evaluate(pi).reshape(4, 4))
''','''
assert np.array_equal(evaluate(pi), V_star), "Policy iteration should reach the same optimal values as value iteration"
''', "Stop when new_pi == pi"),
F(title='Draw the optimal policy',
text="Arrows show the greedy action in each cell, over a heatmap of V*.",
starter='''
arrows = {"up": (0, -0.3), "right": (0.3, 0), "down": (0, 0.3), "left": (-0.3, 0)}
plt.imshow(V_star.reshape(4, 4), cmap="viridis")
for s, a in policy.items():
    r, c = divmod(s, 4); dx, dy = arrows[a]
    plt.arrow(c - dx / 2, r - dy / 2, dx, dy, head_width=0.12, color="white")
plt.colorbar(label="V*"); plt.xticks([]); plt.yticks([])
''', solution='''
arrows = {"up": (0, -0.3), "right": (0.3, 0), "down": (0, 0.3), "left": (-0.3, 0)}
plt.imshow(V_star.reshape(4, 4), cmap="viridis")
for s, a in policy.items():
    r, c = divmod(s, 4); dx, dy = arrows[a]
    plt.arrow(c - dx / 2, r - dy / 2, dx, dy, head_width=0.12, color="white")
plt.colorbar(label="V*"); plt.xticks([]); plt.yticks([])
''', check="assert len(plt.gcf().axes) >= 1", hint="Nothing to fill in: run it."),
]),

dict(id='P11', title='Tabular control: Cliff Walking and Blackjack', packages=['numpy', 'matplotlib'], intro=
"SARSA vs Q-learning on Cliff Walking (Sutton & Barto Figure 6.4), then Monte Carlo control on Blackjack. Both environments are written out for you in plain Python.",
steps=[
F(title='The Cliff Walking environment',
text="A 4x12 grid. Start bottom-left, goal bottom-right, and the cells between them are a cliff: stepping in costs -100 and sends you back to the start. Every other step costs -1. Read it and run it.",
starter='''
import numpy as np
import matplotlib.pyplot as plt
ROWS, COLS, START, GOAL = 4, 12, (3, 0), (3, 11)
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]          # up, right, down, left

def cliff_step(state, a):
    r, c = state
    dr, dc = MOVES[a]
    r, c = min(max(r + dr, 0), ROWS - 1), min(max(c + dc, 0), COLS - 1)
    if r == 3 and 0 < c < 11:
        return START, -100, False
    return (r, c), -1, (r, c) == GOAL

print(cliff_step(START, 1), cliff_step((2, 11), 2))
''', solution='''
import numpy as np
import matplotlib.pyplot as plt
ROWS, COLS, START, GOAL = 4, 12, (3, 0), (3, 11)
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]          # up, right, down, left

def cliff_step(state, a):
    r, c = state
    dr, dc = MOVES[a]
    r, c = min(max(r + dr, 0), ROWS - 1), min(max(c + dc, 0), COLS - 1)
    if r == 3 and 0 < c < 11:
        return START, -100, False
    return (r, c), -1, (r, c) == GOAL

print(cliff_step(START, 1), cliff_step((2, 11), 2))
''', check='''
assert cliff_step(START, 1) == (START, -100, False) and cliff_step((2, 11), 2) == (GOAL, -1, True)
''', hint="Nothing to fill in: read it and run it."),
S('SARSA vs Q-learning',
"Both update Q(s, a) toward r + Q(next). SARSA uses the action it will actually take next (on-policy); Q-learning uses the best next action (off-policy). That one difference changes the route they learn.",
'''
def train(method, episodes=500, alpha=0.5, eps=0.1, seed=1):
    rng = np.random.default_rng(seed)
    Q = np.zeros((ROWS, COLS, 4)); totals = []
    pick = lambda s: rng.integers(4) if rng.random() < eps else int(np.argmax(Q[s]))
    for _ in range(episodes):
        s, a, total, done = START, pick(START), 0, False
        while not done:
            s2, r, done = cliff_step(s, a)
            a2 = pick(s2)
            if method == "sarsa":
                target = r + (0 if done else Q[s2][a2])
            else:
                target = r + (0 if done else ___)
            Q[s][a] += alpha * (target - Q[s][a])
            s, a, total = s2, a2, total + r
        totals.append(total)
    return Q, np.array(totals)

Q_sarsa, sarsa_tot = train("sarsa")
Q_q, q_tot = train("q")
smooth = lambda x: np.convolve(x, np.ones(20) / 20, mode="valid")
plt.plot(smooth(sarsa_tot), label="SARSA"); plt.plot(smooth(q_tot), label="Q-learning")
plt.ylim(-100, 0); plt.xlabel("episode"); plt.ylabel("reward per episode"); plt.legend()
print(sarsa_tot[-100:].mean(), q_tot[-100:].mean())
''','''
def train(method, episodes=500, alpha=0.5, eps=0.1, seed=1):
    rng = np.random.default_rng(seed)
    Q = np.zeros((ROWS, COLS, 4)); totals = []
    pick = lambda s: rng.integers(4) if rng.random() < eps else int(np.argmax(Q[s]))
    for _ in range(episodes):
        s, a, total, done = START, pick(START), 0, False
        while not done:
            s2, r, done = cliff_step(s, a)
            a2 = pick(s2)
            if method == "sarsa":
                target = r + (0 if done else Q[s2][a2])
            else:
                target = r + (0 if done else np.max(Q[s2]))
            Q[s][a] += alpha * (target - Q[s][a])
            s, a, total = s2, a2, total + r
        totals.append(total)
    return Q, np.array(totals)

Q_sarsa, sarsa_tot = train("sarsa")
Q_q, q_tot = train("q")
smooth = lambda x: np.convolve(x, np.ones(20) / 20, mode="valid")
plt.plot(smooth(sarsa_tot), label="SARSA"); plt.plot(smooth(q_tot), label="Q-learning")
plt.ylim(-100, 0); plt.xlabel("episode"); plt.ylabel("reward per episode"); plt.legend()
print(sarsa_tot[-100:].mean(), q_tot[-100:].mean())
''','''
assert sarsa_tot[-100:].mean() > q_tot[-100:].mean(), "SARSA should collect more reward while exploring (Figure 6.4)"
''', "Q-learning looks at the best next action: np.max(Q[s2])"),
S('Which route did each learn?',
"Follow each greedy policy from the start. Q-learning learns the optimal path right along the cliff edge (13 steps); SARSA, which accounts for its own exploration, takes the safer, longer route.",
'''
def greedy_path_length(Q, limit=100):
    s, n = START, 0
    while s != GOAL and n < limit:
        s, _, _ = cliff_step(s, int(np.argmax(___)))
        n += 1
    return n

print(greedy_path_length(Q_q), greedy_path_length(Q_sarsa))
''','''
def greedy_path_length(Q, limit=100):
    s, n = START, 0
    while s != GOAL and n < limit:
        s, _, _ = cliff_step(s, int(np.argmax(Q[s])))
        n += 1
    return n

print(greedy_path_length(Q_q), greedy_path_length(Q_sarsa))
''','''
assert greedy_path_length(Q_q) == 13, "Q-learning's greedy path should be the 13-step cliff-edge route"
assert 13 < greedy_path_length(Q_sarsa) < 40, "SARSA's greedy path should reach the goal by a longer, safer route"
''', "Act greedily: np.argmax(Q[s])"),
S('Monte Carlo control on Blackjack',
"No model, no bootstrapping: play whole hands, then average the returns seen after each (state, action). State = (your sum, dealer's showing card, usable ace). Fill in the return update.",
'''
rng = np.random.default_rng(0)
def draw(): return min(int(rng.integers(1, 14)), 10)
def hand_value(cards):
    s = sum(cards); ace = 1 in cards and s + 10 <= 21
    return (s + 10 if ace else s), ace

def play_hand(policy, eps):
    player, dealer = [draw(), draw()], [draw(), draw()]
    visited = []
    while True:
        total, ace = hand_value(player)
        if total > 21:
            return visited, -1
        state = (total, dealer[0], ace)
        a = int(rng.integers(2)) if rng.random() < eps else policy.get(state, 1 if total < 20 else 0)
        visited.append((state, a))
        if a == 0:
            break
        player.append(draw())
    while hand_value(dealer)[0] < 17:
        dealer.append(draw())
    p, d = hand_value(player)[0], hand_value(dealer)[0]
    return visited, 1 if d > 21 or p > d else (0 if p == d else -1)

Q, counts, policy = {}, {}, {}
for episode in range(100000):
    visited, G = play_hand(policy, eps=0.1)
    for sa in set(visited):
        counts[sa] = counts.get(sa, 0) + 1
        Q[sa] = Q.get(sa, 0.0) + ___
        state = sa[0]
        policy[state] = int(Q.get((state, 1), -9) > Q.get((state, 0), -9))
print(len(Q))
''','''
rng = np.random.default_rng(0)
def draw(): return min(int(rng.integers(1, 14)), 10)
def hand_value(cards):
    s = sum(cards); ace = 1 in cards and s + 10 <= 21
    return (s + 10 if ace else s), ace

def play_hand(policy, eps):
    player, dealer = [draw(), draw()], [draw(), draw()]
    visited = []
    while True:
        total, ace = hand_value(player)
        if total > 21:
            return visited, -1
        state = (total, dealer[0], ace)
        a = int(rng.integers(2)) if rng.random() < eps else policy.get(state, 1 if total < 20 else 0)
        visited.append((state, a))
        if a == 0:
            break
        player.append(draw())
    while hand_value(dealer)[0] < 17:
        dealer.append(draw())
    p, d = hand_value(player)[0], hand_value(dealer)[0]
    return visited, 1 if d > 21 or p > d else (0 if p == d else -1)

Q, counts, policy = {}, {}, {}
for episode in range(100000):
    visited, G = play_hand(policy, eps=0.1)
    for sa in set(visited):
        counts[sa] = counts.get(sa, 0) + 1
        Q[sa] = Q.get(sa, 0.0) + (G - Q.get(sa, 0.0)) / counts[sa]
        state = sa[0]
        policy[state] = int(Q.get((state, 1), -9) > Q.get((state, 0), -9))
print(len(Q))
''','''
stick20 = np.mean([policy.get((20, d, False), 1) == 0 for d in range(1, 11)])
hit_low = np.mean([policy.get((s, d, False), 0) == 1 for s in range(4, 12) for d in range(1, 11)])
assert stick20 >= 0.9 and hit_low >= 0.75, f"The learned policy should stick on 20 and mostly hit on 11 or less; you have {stick20:.0%} and {hit_low:.0%}"
''', "Running average again: (G - Q.get(sa, 0.0)) / counts[sa]"),
F(title='Look at the learned Blackjack policy',
text="Rows are your sum (12-21), columns the dealer's card (ace to 10), without a usable ace. Dark means hit. Compare with Sutton & Barto Figure 5.2: hit on low sums, stick from about 13-17 against weak dealer cards.",
starter='''
grid = np.array([[policy.get((s, d, False), 0) for d in range(1, 11)] for s in range(21, 11, -1)])
plt.imshow(grid, cmap="Greys", extent=[0.5, 10.5, 11.5, 21.5])
plt.xlabel("dealer showing"); plt.ylabel("player sum"); plt.title("dark = hit")
''', solution='''
grid = np.array([[policy.get((s, d, False), 0) for d in range(1, 11)] for s in range(21, 11, -1)])
plt.imshow(grid, cmap="Greys", extent=[0.5, 10.5, 11.5, 21.5])
plt.xlabel("dealer showing"); plt.ylabel("player sum"); plt.title("dark = hit")
''', check="assert grid.shape == (10, 10)", hint="Nothing to fill in: run it."),
]),

dict(id='P12', title='Function approximation: Mountain Car with tile coding', packages=['numpy', 'matplotlib'], intro=
"A car must rock back and forth to climb out of a valley. The state is continuous, so a table won't do: you'll use tile coding (overlapping grids) as features and semi-gradient SARSA, following Sutton & Barto Chapter 10.",
steps=[
S('The physics',
"velocity += 0.001 a - 0.0025 cos(3 x), clipped to [-0.07, 0.07]; position += velocity, clipped to [-1.2, 0.5]. Actions a are -1, 0, +1 (push left, nothing, push right). Reaching x = 0.5 ends the episode.",
'''
import numpy as np
import matplotlib.pyplot as plt

def mc_step(x, v, a):
    v = np.clip(v + 0.001 * a - 0.0025 * np.cos(3 * x), -0.07, 0.07)
    x = ___
    if x == -1.2:
        v = 0.0
    return x, v, x >= 0.5

print(mc_step(-0.5, 0.0, 1))
''','''
import numpy as np
import matplotlib.pyplot as plt

def mc_step(x, v, a):
    v = np.clip(v + 0.001 * a - 0.0025 * np.cos(3 * x), -0.07, 0.07)
    x = np.clip(x + v, -1.2, 0.5)
    if x == -1.2:
        v = 0.0
    return x, v, x >= 0.5

print(mc_step(-0.5, 0.0, 1))
''','''
x, v, done = mc_step(-0.5, 0.0, 1)
assert abs(v - (0.001 - 0.0025 * np.cos(-1.5))) < 1e-12 and abs(x - (-0.5 + v)) < 1e-12 and not done
''', "np.clip(x + v, -1.2, 0.5)"),
S('Tile coding',
"Cover the (position, velocity) square with 8 grids of 8x8 tiles, each shifted by a fraction of a tile. A state lights up exactly one tile per grid, so it has 8 active features. Nearby states share tiles, which is how learning generalises.",
'''
N_TILINGS, N_TILES = 8, 8
def active_tiles(x, v, a):
    xs = (x + 1.2) / 1.7 * N_TILES
    vs = (v + 0.07) / 0.14 * N_TILES
    idx = []
    for t in range(N_TILINGS):
        off = t / N_TILINGS
        i = min(int(xs + off), N_TILES)
        j = min(int(vs + off * 3 % 1), N_TILES)
        idx.append(((t * (N_TILES + 1) + i) * (N_TILES + 1) + j) * 3 + ___)
    return idx

N_FEATURES = N_TILINGS * (N_TILES + 1) ** 2 * 3
print(active_tiles(-0.5, 0.0, 0), N_FEATURES)
''','''
N_TILINGS, N_TILES = 8, 8
def active_tiles(x, v, a):
    xs = (x + 1.2) / 1.7 * N_TILES
    vs = (v + 0.07) / 0.14 * N_TILES
    idx = []
    for t in range(N_TILINGS):
        off = t / N_TILINGS
        i = min(int(xs + off), N_TILES)
        j = min(int(vs + off * 3 % 1), N_TILES)
        idx.append(((t * (N_TILES + 1) + i) * (N_TILES + 1) + j) * 3 + (a + 1))
    return idx

N_FEATURES = N_TILINGS * (N_TILES + 1) ** 2 * 3
print(active_tiles(-0.5, 0.0, 0), N_FEATURES)
''','''
tl = active_tiles(-0.5, 0.0, 1)
assert len(set(tl)) == 8 and all(0 <= i < N_FEATURES for i in tl), "8 distinct, in-range features"
assert set(active_tiles(-0.5, 0.0, -1)).isdisjoint(tl), "Different actions must use different features"
''', "Actions -1, 0, 1 become offsets 0, 1, 2: (a + 1)"),
S('Semi-gradient SARSA',
"q(s, a) is the sum of the weights of the active tiles. Update: w[active] += alpha (r + q(s', a') - q(s, a)). With weights starting at 0 and rewards of -1, the agent is optimistic and explores without epsilon. This takes about 20-40 seconds in the browser.",
'''
rng = np.random.default_rng(0)
w = np.zeros(N_FEATURES)
alpha = 0.5 / N_TILINGS
q = lambda x, v, a: w[active_tiles(x, v, a)].sum()
greedy = lambda x, v: max((-1, 0, 1), key=lambda a: q(x, v, a) + 1e-9 * rng.random())
steps_per_episode = []
for ep in range(120):
    x, v = rng.uniform(-0.6, -0.4), 0.0
    a, n = greedy(x, v), 0
    while True:
        x2, v2, done = mc_step(x, v, a)
        tiles = active_tiles(x, v, a)
        n += 1
        if done or n >= 2000:
            w[tiles] += alpha * (-1 - w[tiles].sum())
            break
        a2 = greedy(x2, v2)
        w[tiles] += alpha * (___ - w[tiles].sum())
        x, v, a = x2, v2, a2
    steps_per_episode.append(n)
print(steps_per_episode[:5], np.mean(steps_per_episode[-20:]))
''','''
rng = np.random.default_rng(0)
w = np.zeros(N_FEATURES)
alpha = 0.5 / N_TILINGS
q = lambda x, v, a: w[active_tiles(x, v, a)].sum()
greedy = lambda x, v: max((-1, 0, 1), key=lambda a: q(x, v, a) + 1e-9 * rng.random())
steps_per_episode = []
for ep in range(120):
    x, v = rng.uniform(-0.6, -0.4), 0.0
    a, n = greedy(x, v), 0
    while True:
        x2, v2, done = mc_step(x, v, a)
        tiles = active_tiles(x, v, a)
        n += 1
        if done or n >= 2000:
            w[tiles] += alpha * (-1 - w[tiles].sum())
            break
        a2 = greedy(x2, v2)
        w[tiles] += alpha * (-1 + q(x2, v2, a2) - w[tiles].sum())
        x, v, a = x2, v2, a2
    steps_per_episode.append(n)
print(steps_per_episode[:5], np.mean(steps_per_episode[-20:]))
''','''
assert np.mean(steps_per_episode[-20:]) < 300, f"After training the car should escape in well under 300 steps; you average {np.mean(steps_per_episode[-20:]):.0f}"
''', "The target is reward plus the next value: -1 + q(x2, v2, a2)"),
F(title='Learning curve and the cost-to-go',
text="Left: steps per episode. Right: -max_a q(x, v, a) over the state space (Sutton & Barto Figure 10.1): the spiral is the rocking strategy.",
starter='''
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(steps_per_episode); axes[0].set_yscale("log"); axes[0].set_xlabel("episode"); axes[0].set_ylabel("steps")
xs, vs = np.linspace(-1.2, 0.5, 40), np.linspace(-0.07, 0.07, 40)
cost = np.array([[-max(q(x, v, a) for a in (-1, 0, 1)) for x in xs] for v in vs])
im = axes[1].imshow(cost, origin="lower", aspect="auto", extent=[-1.2, 0.5, -0.07, 0.07]); axes[1].set_xlabel("position"); axes[1].set_ylabel("velocity")
fig.colorbar(im, ax=axes[1], label="cost to go")
''', solution='''
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(steps_per_episode); axes[0].set_yscale("log"); axes[0].set_xlabel("episode"); axes[0].set_ylabel("steps")
xs, vs = np.linspace(-1.2, 0.5, 40), np.linspace(-0.07, 0.07, 40)
cost = np.array([[-max(q(x, v, a) for a in (-1, 0, 1)) for x in xs] for v in vs])
im = axes[1].imshow(cost, origin="lower", aspect="auto", extent=[-1.2, 0.5, -0.07, 0.07]); axes[1].set_xlabel("position"); axes[1].set_ylabel("velocity")
fig.colorbar(im, ax=axes[1], label="cost to go")
''', check="assert cost.shape == (40, 40)", hint="Nothing to fill in: run it."),
]),

dict(id='P14', title='Policy gradients on CartPole', packages=['numpy', 'matplotlib'], intro=
"Learn a policy directly, with no value table: REINFORCE nudges the policy toward actions that led to high returns. You'll balance a pole using a linear softmax policy, then add a baseline and an actor-critic. CartPole's physics are written out so it runs in your browser.",
steps=[
F(title='The CartPole environment',
text="The standard CartPole equations (the same as Gymnasium's CartPole-v1). The episode ends when the pole tips past 12 degrees, the cart leaves the track, or after 500 steps. A random policy lasts about 20 steps.",
starter='''
import numpy as np
import matplotlib.pyplot as plt

class CartPole:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
    def reset(self):
        self.s = self.rng.uniform(-0.05, 0.05, 4); self.t = 0
        return self.s.copy()
    def step(self, a):
        x, x_dot, th, th_dot = self.s
        force = 10.0 if a == 1 else -10.0
        cos, sin = np.cos(th), np.sin(th)
        temp = (force + 0.05 * th_dot ** 2 * sin) / 1.1
        th_acc = (9.8 * sin - cos * temp) / (0.5 * (4.0 / 3.0 - 0.1 * cos ** 2 / 1.1))
        x_acc = temp - 0.05 * th_acc * cos / 1.1
        x, x_dot = x + 0.02 * x_dot, x_dot + 0.02 * x_acc
        th, th_dot = th + 0.02 * th_dot, th_dot + 0.02 * th_acc
        self.s = np.array([x, x_dot, th, th_dot]); self.t += 1
        done = abs(x) > 2.4 or abs(th) > 0.2095 or self.t >= 500
        return self.s.copy(), 1.0, done

env = CartPole()
rng = np.random.default_rng(1)
lengths = []
for _ in range(50):
    env.reset(); done = False; n = 0
    while not done:
        _, _, done = env.step(rng.integers(2)); n += 1
    lengths.append(n)
print(np.mean(lengths))
''', solution='''
import numpy as np
import matplotlib.pyplot as plt

class CartPole:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
    def reset(self):
        self.s = self.rng.uniform(-0.05, 0.05, 4); self.t = 0
        return self.s.copy()
    def step(self, a):
        x, x_dot, th, th_dot = self.s
        force = 10.0 if a == 1 else -10.0
        cos, sin = np.cos(th), np.sin(th)
        temp = (force + 0.05 * th_dot ** 2 * sin) / 1.1
        th_acc = (9.8 * sin - cos * temp) / (0.5 * (4.0 / 3.0 - 0.1 * cos ** 2 / 1.1))
        x_acc = temp - 0.05 * th_acc * cos / 1.1
        x, x_dot = x + 0.02 * x_dot, x_dot + 0.02 * x_acc
        th, th_dot = th + 0.02 * th_dot, th_dot + 0.02 * th_acc
        self.s = np.array([x, x_dot, th, th_dot]); self.t += 1
        done = abs(x) > 2.4 or abs(th) > 0.2095 or self.t >= 500
        return self.s.copy(), 1.0, done

env = CartPole()
rng = np.random.default_rng(1)
lengths = []
for _ in range(50):
    env.reset(); done = False; n = 0
    while not done:
        _, _, done = env.step(rng.integers(2)); n += 1
    lengths.append(n)
print(np.mean(lengths))
''', check="assert 10 < np.mean(lengths) < 40, 'A random policy should last about 20 steps'", hint="Nothing to fill in: read it and run it."),
S('A softmax policy and its score function',
"The policy is pi(a | s) = softmax(theta^T x)[a], with x = [s, 1]. REINFORCE needs grad log pi(a | s), which for a linear softmax is x times (onehot(a) - pi). Fill it in; the check compares with a numerical derivative.",
'''
def features(s):
    return np.append(s, 1.0)

def policy(theta, s):
    z = features(s) @ theta
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()

def grad_log_pi(theta, s, a):
    p = policy(theta, s)
    onehot = np.eye(2)[a]
    return ___

theta = np.zeros((5, 2))
print(policy(theta, env.reset()))
''','''
def features(s):
    return np.append(s, 1.0)

def policy(theta, s):
    z = features(s) @ theta
    z = z - z.max()
    p = np.exp(z)
    return p / p.sum()

def grad_log_pi(theta, s, a):
    p = policy(theta, s)
    onehot = np.eye(2)[a]
    return np.outer(features(s), onehot - p)

theta = np.zeros((5, 2))
print(policy(theta, env.reset()))
''','''
r = np.random.default_rng(0); th = r.normal(size=(5, 2)); s0 = r.normal(size=4); h = 1e-6
E = np.zeros_like(th); E[2, 1] = h
num = (np.log(policy(th + E, s0)[0]) - np.log(policy(th - E, s0)[0])) / (2 * h)
g = grad_log_pi(th, s0, 0)
assert g.shape == (5, 2) and abs(g[2, 1] - num) < 1e-6, "grad log pi should be np.outer(features(s), onehot - p)"
''', "np.outer(features(s), onehot - p)"),
S('REINFORCE',
"Play an episode, compute the return-to-go G_t at each step (discount 0.99), normalise them, and step theta += lr * sum_t G_t grad log pi(a_t | s_t). This takes about 20-30 seconds. Policy gradients are noisy: your curve will wobble, and rerunning gives a different one. That's normal.",
'''
def run_episode(theta, rng):
    s = env.reset(); states, actions, done = [], [], False
    while not done:
        a = int(rng.random() < policy(theta, s)[1])
        states.append(s); actions.append(a)
        s, _, done = env.step(a)
    return states, actions

def returns_to_go(T, gamma=0.99):
    G, out = 0.0, np.zeros(T)
    for t in reversed(range(T)):
        G = 1.0 + gamma * G
        out[t] = G
    return out

def reinforce(episodes=400, lr=0.01, seed=0, baseline=None):
    rng = np.random.default_rng(seed)
    theta, lengths = np.zeros((5, 2)), []
    for _ in range(episodes):
        states, actions = run_episode(theta, rng)
        G = returns_to_go(len(states))
        adv = baseline(states, G) if baseline else (G - G.mean()) / (G.std() + 1e-8)
        grad = sum(adv[t] * grad_log_pi(theta, states[t], actions[t]) for t in range(len(states)))
        theta += ___
        lengths.append(len(states))
    return theta, lengths

theta, lengths = reinforce()
print(np.mean(lengths[:20]), np.mean(lengths[-20:]))
''','''
def run_episode(theta, rng):
    s = env.reset(); states, actions, done = [], [], False
    while not done:
        a = int(rng.random() < policy(theta, s)[1])
        states.append(s); actions.append(a)
        s, _, done = env.step(a)
    return states, actions

def returns_to_go(T, gamma=0.99):
    G, out = 0.0, np.zeros(T)
    for t in reversed(range(T)):
        G = 1.0 + gamma * G
        out[t] = G
    return out

def reinforce(episodes=400, lr=0.01, seed=0, baseline=None):
    rng = np.random.default_rng(seed)
    theta, lengths = np.zeros((5, 2)), []
    for _ in range(episodes):
        states, actions = run_episode(theta, rng)
        G = returns_to_go(len(states))
        adv = baseline(states, G) if baseline else (G - G.mean()) / (G.std() + 1e-8)
        grad = sum(adv[t] * grad_log_pi(theta, states[t], actions[t]) for t in range(len(states)))
        theta += lr * grad
        lengths.append(len(states))
    return theta, lengths

theta, lengths = reinforce()
print(np.mean(lengths[:20]), np.mean(lengths[-20:]))
''','''
assert np.mean(lengths[-50:]) > 100, f"The pole should stay up far longer than a random policy's ~20 steps; your last-50 average is {np.mean(lengths[-50:]):.0f}"
''', "Gradient ascent: lr * grad"),
S('Add a learned baseline',
"Subtracting a baseline b(s) from the return lowers variance without biasing the gradient. Use a linear value estimate V(s) = w . x, fitted by a small gradient step each episode; the advantage is G_t - V(s_t), then normalised.",
'''
w_v = np.zeros(5)
def value_baseline(states, G, lr_v=0.005):
    global w_v
    X = np.array([features(s) for s in states])
    adv = G - X @ w_v
    w_v += lr_v * X.T @ ___ / len(G)          # fit V toward G
    return (adv - adv.mean()) / (adv.std() + 1e-8)

theta_b, lengths_b = reinforce(baseline=value_baseline)
smooth = lambda x: np.convolve(x, np.ones(20) / 20, mode="valid")
plt.plot(smooth(lengths), label="REINFORCE"); plt.plot(smooth(lengths_b), label="REINFORCE + baseline")
plt.xlabel("episode"); plt.ylabel("steps balanced"); plt.legend()
print(np.mean(lengths_b[-20:]))
''','''
w_v = np.zeros(5)
def value_baseline(states, G, lr_v=0.005):
    global w_v
    X = np.array([features(s) for s in states])
    adv = G - X @ w_v
    w_v += lr_v * X.T @ adv / len(G)          # fit V toward G
    return (adv - adv.mean()) / (adv.std() + 1e-8)

theta_b, lengths_b = reinforce(baseline=value_baseline)
smooth = lambda x: np.convolve(x, np.ones(20) / 20, mode="valid")
plt.plot(smooth(lengths), label="REINFORCE"); plt.plot(smooth(lengths_b), label="REINFORCE + baseline")
plt.xlabel("episode"); plt.ylabel("steps balanced"); plt.legend()
print(np.mean(lengths_b[-20:]))
''','''
assert np.mean(lengths_b[-50:]) > 100, "With a baseline the agent should also learn to balance"
assert np.linalg.norm(w_v) > 0, "The value weights should have been updated"
''', "The regression error is the advantage itself: adv"),
]),

dict(id='P15', title='Options in the Four Rooms', packages=['numpy', 'matplotlib'], intro=
"Hierarchical RL in one picture: in Sutton, Precup & Singh's Four Rooms, an agent that can choose 'go to the hallway' as a single action (an option) learns far faster than one using only up/down/left/right. You'll implement SMDP Q-learning over options.",
steps=[
F(title='The Four Rooms world and its hallway options',
text="An 11x11 grid split into four rooms joined by four hallways. Each option drives the agent to one of its room's two hallways using a shortest-path policy computed by breadth-first search. Read it and run it.",
starter='''
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
layout = """
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
""".strip().split("\\n")
FREE = {(r, c) for r, row in enumerate(layout) for c, ch in enumerate(row) if ch != "w"}
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GOAL, START = (9, 9), (2, 2)
HALLS = [(3, 6), (6, 2), (7, 9), (10, 6)]

def move(s, a, rng):
    if rng.random() < 1 / 3:
        a = rng.integers(4)                       # slippery: a third of moves go a random way
    nxt = (s[0] + MOVES[a][0], s[1] + MOVES[a][1])
    return nxt if nxt in FREE else s

def find_rooms():
    rest, room, rid = FREE - set(HALLS), {}, 0
    for start in sorted(rest):
        if start in room:
            continue
        room[start], q = rid, deque([start])
        while q:
            x = q.popleft()
            for dr, dc in MOVES:
                n = (x[0] + dr, x[1] + dc)
                if n in rest and n not in room:
                    room[n] = rid; q.append(n)
        rid += 1
    return room
ROOM = find_rooms()

def bfs_policy(target, allowed):
    dist, q = {target: 0}, deque([target])
    while q:
        s = q.popleft()
        for dr, dc in MOVES:
            n = (s[0] + dr, s[1] + dc)
            if n in allowed and n not in dist:
                dist[n] = dist[s] + 1; q.append(n)
    return {s: int(np.argmin([dist.get((s[0] + dr, s[1] + dc), 999) for dr, dc in MOVES])) for s in allowed if s != target}

OPTIONS = []
for h in HALLS:
    for rid in {ROOM[(h[0] + dr, h[1] + dc)] for dr, dc in MOVES if (h[0] + dr, h[1] + dc) in ROOM}:
        cells = {x for x in ROOM if ROOM[x] == rid}
        OPTIONS.append({"target": h, "init": cells, "pi": bfs_policy(h, cells | {h})})
print(len(FREE), len(OPTIONS))
''', solution='''
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
layout = """
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
""".strip().split("\\n")
FREE = {(r, c) for r, row in enumerate(layout) for c, ch in enumerate(row) if ch != "w"}
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GOAL, START = (9, 9), (2, 2)
HALLS = [(3, 6), (6, 2), (7, 9), (10, 6)]

def move(s, a, rng):
    if rng.random() < 1 / 3:
        a = rng.integers(4)                       # slippery: a third of moves go a random way
    nxt = (s[0] + MOVES[a][0], s[1] + MOVES[a][1])
    return nxt if nxt in FREE else s

def find_rooms():
    rest, room, rid = FREE - set(HALLS), {}, 0
    for start in sorted(rest):
        if start in room:
            continue
        room[start], q = rid, deque([start])
        while q:
            x = q.popleft()
            for dr, dc in MOVES:
                n = (x[0] + dr, x[1] + dc)
                if n in rest and n not in room:
                    room[n] = rid; q.append(n)
        rid += 1
    return room
ROOM = find_rooms()

def bfs_policy(target, allowed):
    dist, q = {target: 0}, deque([target])
    while q:
        s = q.popleft()
        for dr, dc in MOVES:
            n = (s[0] + dr, s[1] + dc)
            if n in allowed and n not in dist:
                dist[n] = dist[s] + 1; q.append(n)
    return {s: int(np.argmin([dist.get((s[0] + dr, s[1] + dc), 999) for dr, dc in MOVES])) for s in allowed if s != target}

OPTIONS = []
for h in HALLS:
    for rid in {ROOM[(h[0] + dr, h[1] + dc)] for dr, dc in MOVES if (h[0] + dr, h[1] + dc) in ROOM}:
        cells = {x for x in ROOM if ROOM[x] == rid}
        OPTIONS.append({"target": h, "init": cells, "pi": bfs_policy(h, cells | {h})})
print(len(FREE), len(OPTIONS))
''', check="assert len(OPTIONS) == 8, 'Each of the 4 hallways is reachable from 2 rooms: 8 options'", hint="Nothing to fill in: read it and run it."),
S('SMDP Q-learning',
"Choices are the 4 primitive moves plus any option available in the current cell. An option runs until it reaches its hallway (or the goal), taking k steps and collecting discounted reward R. The SMDP update discounts the next state's value by gamma^k instead of gamma.",
'''
gamma, alpha, eps = 0.9, 0.25, 0.1

def choices(s, use_options):
    base = [("prim", a) for a in range(4)]
    return base + ([("opt", i) for i, o in enumerate(OPTIONS) if s in o["init"]] if use_options else [])

def execute(s, choice, rng):
    kind, i = choice
    if kind == "prim":
        s2 = move(s, i, rng)
        return s2, (1.0 if s2 == GOAL else 0.0), 1
    o, R, k = OPTIONS[i], 0.0, 0
    while True:
        s = move(s, o["pi"][s], rng)
        R += gamma ** k * (1.0 if s == GOAL else 0.0); k += 1
        if s == GOAL or s == o["target"] or s not in o["pi"]:
            return s, R, k

def learn(use_options, episodes=80, seed=0):
    rng = np.random.default_rng(seed)
    Q, steps = {}, []
    for _ in range(episodes):
        s, total = START, 0
        while s != GOAL and total < 5000:
            cs = choices(s, use_options)
            c = cs[rng.integers(len(cs))] if rng.random() < eps else max(cs, key=lambda c: Q.get((s, c), 0.0) + 1e-9 * rng.random())
            s2, R, k = execute(s, c, rng)
            best_next = 0.0 if s2 == GOAL else max(Q.get((s2, c2), 0.0) for c2 in choices(s2, use_options))
            target = R + ___ * best_next
            Q[(s, c)] = Q.get((s, c), 0.0) + alpha * (target - Q.get((s, c), 0.0))
            s, total = s2, total + k
        steps.append(total)
    return steps

prim_steps = learn(False)
opt_steps = learn(True)
print(sum(prim_steps), sum(opt_steps))
''','''
gamma, alpha, eps = 0.9, 0.25, 0.1

def choices(s, use_options):
    base = [("prim", a) for a in range(4)]
    return base + ([("opt", i) for i, o in enumerate(OPTIONS) if s in o["init"]] if use_options else [])

def execute(s, choice, rng):
    kind, i = choice
    if kind == "prim":
        s2 = move(s, i, rng)
        return s2, (1.0 if s2 == GOAL else 0.0), 1
    o, R, k = OPTIONS[i], 0.0, 0
    while True:
        s = move(s, o["pi"][s], rng)
        R += gamma ** k * (1.0 if s == GOAL else 0.0); k += 1
        if s == GOAL or s == o["target"] or s not in o["pi"]:
            return s, R, k

def learn(use_options, episodes=80, seed=0):
    rng = np.random.default_rng(seed)
    Q, steps = {}, []
    for _ in range(episodes):
        s, total = START, 0
        while s != GOAL and total < 5000:
            cs = choices(s, use_options)
            c = cs[rng.integers(len(cs))] if rng.random() < eps else max(cs, key=lambda c: Q.get((s, c), 0.0) + 1e-9 * rng.random())
            s2, R, k = execute(s, c, rng)
            best_next = 0.0 if s2 == GOAL else max(Q.get((s2, c2), 0.0) for c2 in choices(s2, use_options))
            target = R + gamma ** k * best_next
            Q[(s, c)] = Q.get((s, c), 0.0) + alpha * (target - Q.get((s, c), 0.0))
            s, total = s2, total + k
        steps.append(total)
    return steps

prim_steps = learn(False)
opt_steps = learn(True)
print(sum(prim_steps), sum(opt_steps))
''','''
assert sum(opt_steps) < 0.7 * sum(prim_steps), "With options the agent should need far fewer total steps to learn"
''', "Discount by the option's duration: gamma ** k"),
F(title='Compare learning speed',
text="Steps to reach the goal per episode (log scale). Options let the agent cross whole rooms in one decision, so it finds the goal quickly from the very first episodes.",
starter='''
smooth = lambda x: np.convolve(x, np.ones(5) / 5, mode="valid")
plt.semilogy(smooth(prim_steps), label="primitive actions only")
plt.semilogy(smooth(opt_steps), label="with hallway options")
plt.xlabel("episode"); plt.ylabel("steps to goal"); plt.legend()
''', solution='''
smooth = lambda x: np.convolve(x, np.ones(5) / 5, mode="valid")
plt.semilogy(smooth(prim_steps), label="primitive actions only")
plt.semilogy(smooth(opt_steps), label="with hallway options")
plt.xlabel("episode"); plt.ylabel("steps to goal"); plt.legend()
''', check="assert len(plt.gca().lines) == 2", hint="Nothing to fill in: run it."),
]),
]
