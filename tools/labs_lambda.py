"""Two warm-up labs that take the fear out of lambda functions."""
from labs_warmup import S

LAMBDA = [
dict(id='WL1', title='Lambda functions from zero', packages=[], intro=
"A lambda is just a function written on one line without a name. That's the whole secret. Nothing a lambda does is new: every lambda can be rewritten as a normal `def`, and you never *have* to use one. This lab builds the idea slowly, side by side with `def`, so the syntax stops looking strange.",
steps=[
S('Functions are values',
"In Python a function is a value, like a number or a string. You can store it in a variable, put it in a list, and pass it to another function. Think of how SQL lets you pass an expression (like `UPPER(name)`) into another clause: same idea.\n\nBelow, `apply_twice` receives a function `f` and calls it twice. Pass it `add_ten` (no brackets: you're handing over the function itself, not calling it).",
'''
def add_ten(x):
    return x + 10

def apply_twice(f, x):
    return f(f(x))

result = apply_twice(___, 5)
print(result)
''','''
def add_ten(x):
    return x + 10

def apply_twice(f, x):
    return f(f(x))

result = apply_twice(add_ten, 5)
print(result)
''','''
assert result == 25, "apply_twice(add_ten, 5) should be 25: 5 -> 15 -> 25. Pass add_ten without brackets."
''', "Write add_ten with no brackets: apply_twice(add_ten, 5)"),
S('Your first lambda: the same function, written shorter',
"These two lines create exactly the same function:\n\n`def double(x): return x * 2`\n\n`double = lambda x: x * 2`\n\nRead `lambda x: x * 2` as \"a function that takes x and gives back x * 2\". The part before the colon is the input; the part after is what comes back. There's no `return` keyword: the expression after the colon IS the return value.\n\nWrite `triple` as a lambda.",
'''
def double_def(x):
    return x * 2

double_lambda = lambda x: x * 2

triple = lambda x: ___
print(double_def(4), double_lambda(4), triple(4))
''','''
def double_def(x):
    return x * 2

double_lambda = lambda x: x * 2

triple = lambda x: x * 3
print(double_def(4), double_lambda(4), triple(4))
''','''
assert double_def(7) == double_lambda(7) == 14
assert triple(4) == 12 and triple(-1) == -3, "triple should be lambda x: x * 3"
''', "Copy the double_lambda line and change 2 to 3"),
S('Translate def into lambda',
"Practise the translation in one direction. For each `def`, the lambda is: `lambda` + the inputs + `:` + whatever comes after `return`.\n\n`def f(a, b): return a + b` becomes `lambda a, b: a + b` (several inputs are separated by commas).\n\n`def g(): return 42` becomes `lambda: 42` (no inputs at all).",
'''
def area(width, height):
    return width * height

def full_name(first, last):
    return first + " " + last

def answer():
    return 42

area_l = lambda ___: ___
full_name_l = lambda ___: ___
answer_l = ___
print(area_l(3, 4), full_name_l("Ada", "Lovelace"), answer_l())
''','''
def area(width, height):
    return width * height

def full_name(first, last):
    return first + " " + last

def answer():
    return 42

area_l = lambda width, height: width * height
full_name_l = lambda first, last: first + " " + last
answer_l = lambda: 42
print(area_l(3, 4), full_name_l("Ada", "Lovelace"), answer_l())
''','''
assert area_l(3, 4) == 12, "area_l should be lambda width, height: width * height"
assert full_name_l("Ada", "Lovelace") == "Ada Lovelace"
assert answer_l() == 42, "answer_l takes no inputs: lambda: 42"
''', "lambda width, height: width * height"),
S('Translate lambda into def',
"Now the other direction, which is what you'll do when you meet a lambda in someone else's code and it looks scary: rewrite it as a `def` in your head (or on paper).\n\n`lambda a, b: a + b` means `def something(a, b): return a + b`.",
'''
discount = lambda price, pct: price * (1 - pct / 100)

def discount_def(price, pct):
    return ___

print(discount(200, 10), discount_def(200, 10))
''','''
discount = lambda price, pct: price * (1 - pct / 100)

def discount_def(price, pct):
    return price * (1 - pct / 100)

print(discount(200, 10), discount_def(200, 10))
''','''
assert discount_def(200, 10) == discount(200, 10) == 180, "Copy everything after the colon into the return line"
assert discount_def(50, 50) == 25
''', "Everything after the lambda's colon goes after return"),
S('If/else inside a lambda: SQL CASE WHEN',
"A lambda holds one expression, so you can't write a normal if-block in it. Instead use Python's one-line conditional: `A if condition else B`. It reads like SQL's `CASE WHEN condition THEN A ELSE B END`.\n\nWrite `size_label` that returns \"big\" for amounts over 50 and \"small\" otherwise.",
'''
size_label = lambda amount: ___
print([size_label(a) for a in [120, 40, 300, 75, 10]])
''','''
size_label = lambda amount: "big" if amount > 50 else "small"
print([size_label(a) for a in [120, 40, 300, 75, 10]])
''','''
assert [size_label(a) for a in [120, 40, 300, 75, 10]] == ["big", "small", "big", "big", "small"]
assert size_label(50) == "small", "Exactly 50 is not over 50"
''', '"big" if amount > 50 else "small"'),
S('Where lambdas are actually used: key=',
"You'll almost never see `name = lambda ...` in good code (a `def` is clearer). Lambdas mostly appear in ONE place: as a throwaway function handed to `sorted`, `max`, `min`, which ask \"what should I sort / compare by?\" through `key=`.\n\n`sorted(rows, key=f)` sorts rows by `f(row)`: it's `ORDER BY f(row)`. Sort these (branch, amount) rows by amount. A row is a tuple, so `row[1]` is the amount.",
'''
rows = [("HYD", 130), ("BLR", 70), ("MUM", 250), ("DEL", 90)]

def get_amount(row):
    return row[1]

by_amount_def = sorted(rows, key=get_amount)
by_amount_lambda = sorted(rows, key=lambda row: ___)
print(by_amount_def)
print(by_amount_lambda)
''','''
rows = [("HYD", 130), ("BLR", 70), ("MUM", 250), ("DEL", 90)]

def get_amount(row):
    return row[1]

by_amount_def = sorted(rows, key=get_amount)
by_amount_lambda = sorted(rows, key=lambda row: row[1])
print(by_amount_def)
print(by_amount_lambda)
''','''
assert by_amount_lambda == by_amount_def == [("BLR", 70), ("DEL", 90), ("HYD", 130), ("MUM", 250)], "key=lambda row: row[1]"
''', "The lambda does exactly what get_amount does: lambda row: row[1]"),
S('ORDER BY ... DESC, and ORDER BY two columns',
"`reverse=True` gives DESC. To sort by two things, return a tuple: Python compares tuples left to right, exactly like `ORDER BY city, amount`. A minus sign flips one numeric column to descending.",
'''
txns = [("HYD", 130), ("BLR", 70), ("HYD", 40), ("BLR", 200), ("HYD", 90)]
desc = sorted(txns, key=lambda t: t[1], reverse=___)
by_city_then_amount_desc = sorted(txns, key=lambda t: (t[0], ___))
print(desc)
print(by_city_then_amount_desc)
''','''
txns = [("HYD", 130), ("BLR", 70), ("HYD", 40), ("BLR", 200), ("HYD", 90)]
desc = sorted(txns, key=lambda t: t[1], reverse=True)
by_city_then_amount_desc = sorted(txns, key=lambda t: (t[0], -t[1]))
print(desc)
print(by_city_then_amount_desc)
''','''
assert desc[0] == ("BLR", 200) and desc[-1] == ("HYD", 40)
assert by_city_then_amount_desc == [("BLR", 200), ("BLR", 70), ("HYD", 130), ("HYD", 90), ("HYD", 40)], "Sort by (city, -amount)"
''', "reverse=True; and (t[0], -t[1])"),
S('max and min with key= : the most important pattern in RL',
"`max(items, key=f)` returns the item whose f(item) is largest. It's the argmax: \"which action has the highest Q-value?\". You'll write this line dozens of times in the RL labs:\n\n`best_action = max(actions, key=lambda a: Q[a])`\n\nRead it as: \"among the actions, pick the one with the largest Q[a]\".",
'''
Q = {"up": -3.0, "right": -1.5, "down": -4.2, "left": -2.0}
actions = ["up", "right", "down", "left"]

best_action = max(actions, key=lambda a: ___)
worst_action = min(actions, key=lambda a: Q[a])
print(best_action, worst_action)
''','''
Q = {"up": -3.0, "right": -1.5, "down": -4.2, "left": -2.0}
actions = ["up", "right", "down", "left"]

best_action = max(actions, key=lambda a: Q[a])
worst_action = min(actions, key=lambda a: Q[a])
print(best_action, worst_action)
''','''
assert best_action == "right" and worst_action == "down", "best_action = max(actions, key=lambda a: Q[a])"
''', "Copy the min line's lambda: lambda a: Q[a]"),
S('The same thing without a lambda (always allowed)',
"If a lambda ever feels confusing, write the long version. These are exactly equivalent, and neither is \"more correct\". Rewrite the argmax with a plain loop.",
'''
def argmax_loop(actions, Q):
    best, best_value = None, float("-inf")
    for a in actions:
        if Q[a] > best_value:
            best, best_value = ___, ___
    return best

print(argmax_loop(actions, Q), max(actions, key=lambda a: Q[a]))
''','''
def argmax_loop(actions, Q):
    best, best_value = None, float("-inf")
    for a in actions:
        if Q[a] > best_value:
            best, best_value = a, Q[a]
    return best

print(argmax_loop(actions, Q), max(actions, key=lambda a: Q[a]))
''','''
assert argmax_loop(actions, Q) == "right"
assert argmax_loop(["a", "b"], {"a": 5, "b": 2}) == "a"
''', "Remember the action and its value: a, Q[a]"),
S('Rule of thumb: when to use lambda and when to use def',
"Use a lambda only for a tiny function you pass straight into something else (`key=`, `apply`, `map`). Use `def` when the function has a name you'll reuse, needs more than one expression, or deserves a docstring.\n\nQuiz: for each case, put \"lambda\" or \"def\" in the list.\n\n1. Sort customers by balance.\n2. A reward function with three if-branches and a log statement.\n3. Pick the action with the highest Q-value.\n4. A data-cleaning function used in five different notebooks.",
'''
answers = [___, ___, ___, ___]
''','''
answers = ["lambda", "def", "lambda", "def"]
''','''
assert [a.strip().lower() for a in answers] == ["lambda", "def", "lambda", "def"], "Tiny and passed straight in: lambda. Named, reused or multi-line: def."
''', "Sorting and argmax are one-liners passed to key=; the other two need def"),
]),

dict(id='WL2', title='Lambda drills: reading the lambdas in your course labs', packages=['numpy', 'pandas'], intro=
"Every lambda below is copied (or simplified) from a lab you'll meet later in this plan: grid worlds, bandits, policy gradients, pandas. Do these drills now and those lines will read as plain English when you get there. If any step feels hard, the hint always shows the `def` version.",
steps=[
S('Lambdas in pandas: computed columns',
"`series.map(f)` applies f to every value, like `SELECT f(col)`. `df.apply(f, axis=1)` applies f to every row (each row arrives as a mini-dictionary), which is the pandas version of a computed column that uses several fields.",
'''
import pandas as pd
import numpy as np
df = pd.DataFrame({"city": ["HYD", "BLR", "HYD", "MUM"], "balance": [52000, 18000, 75000, 30000], "loans": [1, 0, 2, 1]})

df["tier"] = df["balance"].map(lambda b: "gold" if b >= 50000 else ___)
df["balance_per_loan"] = df.apply(lambda row: row["balance"] / max(row[___], 1), axis=1)
print(df)
''','''
import pandas as pd
import numpy as np
df = pd.DataFrame({"city": ["HYD", "BLR", "HYD", "MUM"], "balance": [52000, 18000, 75000, 30000], "loans": [1, 0, 2, 1]})

df["tier"] = df["balance"].map(lambda b: "gold" if b >= 50000 else "standard")
df["balance_per_loan"] = df.apply(lambda row: row["balance"] / max(row["loans"], 1), axis=1)
print(df)
''','''
assert list(df["tier"]) == ["gold", "standard", "gold", "standard"], 'Use "standard" for the else branch'
assert list(df["balance_per_loan"]) == [52000, 18000, 37500, 30000], 'Divide by row["loans"] (at least 1)'
''', 'def tier(b): return "gold" if b >= 50000 else "standard". In the row lambda, use row["loans"]'),
S('Lambdas in groupby: custom aggregates',
"`groupby(...).agg(f)` aggregates each group with any function you give it, like a user-defined aggregate in SQL. Compute each city's range (max minus min balance).",
'''
spread = df.groupby("city")["balance"].agg(lambda s: ___)
print(spread)
''','''
spread = df.groupby("city")["balance"].agg(lambda s: s.max() - s.min())
print(spread)
''','''
assert spread["HYD"] == 23000 and spread["BLR"] == 0, "lambda s: s.max() - s.min()"
''', "def spread(s): return s.max() - s.min()"),
S('From the grid-world lab: greedy action selection',
"In the grid-world lab you'll see:\n\n`max(ACTIONS, key=lambda a: -1 + V[next_state(s, a)])`\n\nRead it as: \"try every action a; score it as -1 plus the value of where it takes me; return the best one\". Fill in the score.",
'''
V = np.array([0, -1, -2, -3])               # value of each of 4 cells in a corridor; cell 0 is the goal
def next_state(s, a):
    return max(s - 1, 0) if a == "left" else min(s + 1, 3)
ACTIONS = ["left", "right"]

s = 2
greedy = max(ACTIONS, key=lambda a: ___)
print(greedy)
''','''
V = np.array([0, -1, -2, -3])               # value of each of 4 cells in a corridor; cell 0 is the goal
def next_state(s, a):
    return max(s - 1, 0) if a == "left" else min(s + 1, 3)
ACTIONS = ["left", "right"]

s = 2
greedy = max(ACTIONS, key=lambda a: -1 + V[next_state(s, a)])
print(greedy)
''','''
assert greedy == "left", "Moving left leads toward the goal (higher value)"
''', "def score(a): return -1 + V[next_state(s, a)]"),
S('From the Cliff Walking lab: epsilon-greedy as a lambda',
"The Cliff Walking lab has `pick = lambda s: rng.integers(4) if rng.random() < eps else int(np.argmax(Q[s]))`. It's an if/else lambda: with probability eps pick a random action, otherwise the best one. Rewrite it as a def; the check runs both and compares.",
'''
rng = np.random.default_rng(0)
Q = np.array([[0.1, 0.5, 0.2, 0.0], [0.0, 0.0, 0.9, 0.1]])
eps = 0.0                                   # no exploration, so the result is predictable
pick = lambda s: rng.integers(4) if rng.random() < eps else int(np.argmax(Q[s]))

def pick_def(s):
    if rng.random() < eps:
        return ___
    return ___

print(pick(0), pick_def(0), pick(1), pick_def(1))
''','''
rng = np.random.default_rng(0)
Q = np.array([[0.1, 0.5, 0.2, 0.0], [0.0, 0.0, 0.9, 0.1]])
eps = 0.0                                   # no exploration, so the result is predictable
pick = lambda s: rng.integers(4) if rng.random() < eps else int(np.argmax(Q[s]))

def pick_def(s):
    if rng.random() < eps:
        return rng.integers(4)
    return int(np.argmax(Q[s]))

print(pick(0), pick_def(0), pick(1), pick_def(1))
''','''
assert pick_def(0) == pick(0) == 1 and pick_def(1) == pick(1) == 2
''', "The part before 'if' goes in the first return, the part after 'else' in the second"),
S('From the policy-gradient lab: a one-line helper',
"Labs often define tiny helpers like `smooth = lambda x: np.convolve(x, np.ones(20) / 20, mode=\"valid\")` (a 20-point moving average, like `AVG() OVER (ROWS 19 PRECEDING)`). Write a 3-point version.",
'''
smooth3 = lambda x: ___
print(smooth3(np.array([3.0, 6, 9, 12, 15])))
''','''
smooth3 = lambda x: np.convolve(x, np.ones(3) / 3, mode="valid")
print(smooth3(np.array([3.0, 6, 9, 12, 15])))
''','''
assert np.allclose(smooth3(np.array([3.0, 6, 9, 12, 15])), [6, 9, 12]), 'np.convolve(x, np.ones(3) / 3, mode="valid")'
''', 'def smooth3(x): return np.convolve(x, np.ones(3) / 3, mode="valid")'),
S('From the bandit labs: lambdas with several inputs',
"The bandit labs define strategies like `ucb1 = lambda N, Ssum, t: ...` so they can be swapped in and out of the same runner. Several inputs, one expression. Write a greedy strategy that returns the index of the arm with the highest average reward Ssum / N (assume every arm was pulled at least once).",
'''
greedy_strategy = lambda N, Ssum, t: ___
N = np.array([10, 5, 8]); Ssum = np.array([4.0, 4.0, 2.0])
print(greedy_strategy(N, Ssum, 23))
''','''
greedy_strategy = lambda N, Ssum, t: int(np.argmax(Ssum / N))
N = np.array([10, 5, 8]); Ssum = np.array([4.0, 4.0, 2.0])
print(greedy_strategy(N, Ssum, 23))
''','''
assert greedy_strategy(N, Ssum, 23) == 1, "Averages are 0.4, 0.8, 0.25: arm 1. Use np.argmax(Ssum / N)"
''', "def greedy_strategy(N, Ssum, t): return int(np.argmax(Ssum / N))"),
S('The one trap: lambdas in a loop',
"This is the only genuinely confusing thing about lambdas, so here it is up front. A lambda looks up outside variables when it is CALLED, not when it is created. So lambdas made in a loop all see the loop variable's final value.\n\nThe fix is to freeze the value as a default argument: `lambda i=i: i`. Run it, see the surprise, then fix `fixed`.",
'''
broken = [lambda: i for i in range(3)]
print("broken:", [f() for f in broken])          # surprise: [2, 2, 2]

fixed = [lambda i=___: i for i in range(3)]
print("fixed: ", [f() for f in fixed])
''','''
broken = [lambda: i for i in range(3)]
print("broken:", [f() for f in broken])          # surprise: [2, 2, 2]

fixed = [lambda i=i: i for i in range(3)]
print("fixed: ", [f() for f in fixed])
''','''
assert [f() for f in fixed] == [0, 1, 2], "Freeze the current value with a default: lambda i=i: i"
''', "lambda i=i: i (the default value is captured at creation time)"),
S('Final drill: translate five real lambdas',
"For each lambda from the course labs, write the equivalent `def`. When this passes, you've seen every lambda shape the plan uses.",
'''
f1 = lambda kv: kv[1]
f2 = lambda x, v: max((-1, 0, 1), key=lambda a: x * a + v)
f3 = lambda p: 1 / (1 + np.exp(-p))
f4 = lambda row: row["amount"] * 1.18 if row["taxable"] else row["amount"]
f5 = lambda: np.random.default_rng(0).integers(1, 7)

def g1(kv):
    return ___

def g2(x, v):
    def score(a):
        return ___
    return max((-1, 0, 1), key=score)

def g3(p):
    return ___

def g4(row):
    if row["taxable"]:
        return ___
    return ___

def g5():
    return ___
''','''
f1 = lambda kv: kv[1]
f2 = lambda x, v: max((-1, 0, 1), key=lambda a: x * a + v)
f3 = lambda p: 1 / (1 + np.exp(-p))
f4 = lambda row: row["amount"] * 1.18 if row["taxable"] else row["amount"]
f5 = lambda: np.random.default_rng(0).integers(1, 7)

def g1(kv):
    return kv[1]

def g2(x, v):
    def score(a):
        return x * a + v
    return max((-1, 0, 1), key=score)

def g3(p):
    return 1 / (1 + np.exp(-p))

def g4(row):
    if row["taxable"]:
        return row["amount"] * 1.18
    return row["amount"]

def g5():
    return np.random.default_rng(0).integers(1, 7)
''','''
assert g1(("a", 7)) == f1(("a", 7)) == 7
assert g2(2.0, 0.5) == f2(2.0, 0.5) and g2(-3.0, 1.0) == f2(-3.0, 1.0)
assert abs(g3(0.3) - f3(0.3)) < 1e-12
assert g4({"amount": 100, "taxable": True}) == f4({"amount": 100, "taxable": True}) and g4({"amount": 100, "taxable": False}) == 100
assert g5() == f5()
print("All five translated. Lambdas are just short defs.")
''', "Each body is the text after the lambda's colon. For f4, the part before 'if' is the taxable case."),
]),
]
