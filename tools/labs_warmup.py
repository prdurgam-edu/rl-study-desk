"""Python warm-up labs for someone fluent in SQL who hasn't coded in a while."""

def S(title, text, starter, solution, check, hint):
    return dict(title=title, text=text, starter=starter.strip('\n'), solution=solution.strip('\n'), check=check.strip('\n'), hint=hint)

WARMUP = [
dict(id='W1', title='Python in an hour, for SQL people', packages=[], intro=
"You already think in rows, columns and filters. Python is the same ideas written as steps. Each step below has a few blanks marked ___. Fill them in, press **Run** to see what happens, then **Check**. Stuck for more than 5 minutes? Press **Hint**, then **Solution**. Copying a solution and then retyping it from memory is a perfectly good way to learn.",
steps=[
S('Variables',
"A variable is a name for a value, like a column alias. `=` assigns. `print()` shows a value.\n\nCompute the rejection rate of a data load as a fraction.",
'''
rows_loaded = 1200
rows_rejected = 36

reject_rate = ___
print(reject_rate)
''','''
rows_loaded = 1200
rows_rejected = 36

reject_rate = rows_rejected / rows_loaded
print(reject_rate)
''','''
assert 'reject_rate' in dir(), "Define a variable called reject_rate"
assert abs(reject_rate - 0.03) < 1e-9, f"reject_rate should be 36/1200 = 0.03, you have {reject_rate}"
''', "Divide one variable by the other with /"),
S('Lists are a column',
"A list holds many values in order: `[120, 40, 300]`. Built-in functions work on the whole list: `sum(x)`, `len(x)`, `max(x)`, like SUM, COUNT and MAX.\n\nCompute the average amount.",
'''
amounts = [120, 40, 300, 75, 10]

total = sum(amounts)
count = ___
avg_amount = ___
print(total, count, avg_amount)
''','''
amounts = [120, 40, 300, 75, 10]

total = sum(amounts)
count = len(amounts)
avg_amount = total / count
print(total, count, avg_amount)
''','''
assert count == 5, "count should be len(amounts)"
assert abs(avg_amount - 109) < 1e-9, f"avg_amount should be 109, you have {avg_amount}"
''', "count = len(amounts); avg_amount = total / count"),
S('Loops and if: row by row',
"A `for` loop visits each value, like a cursor. `if` is your WHERE. Indentation (4 spaces) shows what's inside the loop.\n\nCollect the amounts greater than 50 into a new list.",
'''
big = []
for a in amounts:
    if ___:
        big.append(a)
print(big)
''','''
big = []
for a in amounts:
    if a > 50:
        big.append(a)
print(big)
''','''
assert big == [120, 300, 75], f"big should be [120, 300, 75], you have {big}"
''', "The condition is a > 50"),
S('List comprehensions: SELECT ... WHERE in one line',
"`[expression for x in list if condition]` reads almost like SQL: SELECT expression FROM list WHERE condition.\n\nWrite the same filter in one line, and a second list with every amount doubled.",
'''
big2 = [a for a in amounts if ___]
doubled = [___ for a in amounts]
print(big2, doubled)
''','''
big2 = [a for a in amounts if a > 50]
doubled = [a * 2 for a in amounts]
print(big2, doubled)
''','''
assert big2 == [120, 300, 75], "big2 should match big"
assert doubled == [240, 80, 600, 150, 20], f"doubled should be each amount times 2, you have {doubled}"
''', "doubled uses a * 2 as the expression"),
S('Functions: a reusable query',
"`def name(inputs):` defines a function; `return` sends back the answer. Think of it like a stored function.\n\nWrite `pct_change(old, new)` returning the percentage change.",
'''
def pct_change(old, new):
    return ___

print(pct_change(50, 75))
''','''
def pct_change(old, new):
    return (new - old) / old * 100

print(pct_change(50, 75))
''','''
assert abs(pct_change(50, 75) - 50) < 1e-9, "pct_change(50, 75) should be 50"
assert abs(pct_change(200, 150) + 25) < 1e-9, "pct_change(200, 150) should be -25"
''', "(new - old) / old * 100"),
]),

dict(id='W2', title='Dictionaries: GROUP BY by hand', packages=[], intro=
"A dictionary maps keys to values, like a two-column lookup table. It's how you'd do GROUP BY without a database.",
steps=[
S('Dictionary basics',
"`{key: value}` creates one; `d[key]` reads or sets a value.\n\nAdd Mumbai's total and read Hyderabad's.",
'''
branch_totals = {"HYD": 1200, "BLR": 800}
branch_totals["MUM"] = ___
hyd = branch_totals[___]
print(branch_totals, hyd)
''','''
branch_totals = {"HYD": 1200, "BLR": 800}
branch_totals["MUM"] = 1500
hyd = branch_totals["HYD"]
print(branch_totals, hyd)
''','''
assert branch_totals.get("MUM") == 1500, "Set branch_totals['MUM'] to 1500"
assert hyd == 1200, "hyd should be branch_totals['HYD']"
''', 'Keys are strings in quotes: branch_totals["MUM"]'),
S('SUM(amount) GROUP BY branch',
"`d.get(key, 0)` returns the value or 0 if the key is new. Add each amount to its branch's running total.",
'''
txns = [("HYD", 100), ("BLR", 50), ("HYD", 30), ("MUM", 70), ("BLR", 20)]

totals = {}
for branch, amt in txns:
    totals[branch] = totals.get(branch, 0) + ___
print(totals)
''','''
txns = [("HYD", 100), ("BLR", 50), ("HYD", 30), ("MUM", 70), ("BLR", 20)]

totals = {}
for branch, amt in txns:
    totals[branch] = totals.get(branch, 0) + amt
print(totals)
''','''
assert totals == {"HYD": 130, "BLR": 70, "MUM": 70}, f"Expected HYD 130, BLR 70, MUM 70; you have {totals}"
''', "Add amt"),
S('ORDER BY total DESC',
"`sorted(items, key=..., reverse=True)` sorts. `totals.items()` gives (key, value) pairs; `lambda kv: kv[1]` means 'sort by the value'.",
'''
ranked = sorted(totals.items(), key=lambda kv: ___, reverse=True)
top_branch = ranked[0][0]
print(ranked, top_branch)
''','''
ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
top_branch = ranked[0][0]
print(ranked, top_branch)
''','''
assert top_branch == "HYD", "HYD has the largest total"
assert [v for _, v in ranked] == sorted(totals.values(), reverse=True), "ranked should be sorted by total, largest first"
''', "kv[1] is the value"),
S('COUNT(*) GROUP BY as a function',
"Write `group_count(items)` that returns how many times each item appears.",
'''
def group_count(items):
    counts = {}
    for x in items:
        ___
    return counts

print(group_count(["a", "b", "c", "a"]))
''','''
def group_count(items):
    counts = {}
    for x in items:
        counts[x] = counts.get(x, 0) + 1
    return counts

print(group_count(["a", "b", "c", "a"]))
''','''
assert group_count(list("abca")) == {"a": 2, "b": 1, "c": 1}, "group_count(['a','b','c','a']) should be {'a': 2, 'b': 1, 'c': 1}"
assert group_count([]) == {}, "An empty list should give an empty dict"
''', "Same pattern as the previous step, adding 1 instead of an amount"),
]),

dict(id='W3', title='NumPy: math on whole columns', packages=['numpy'], intro=
"NumPy arrays let you do math on a whole column at once, with no loop. Almost all ML code is NumPy underneath.",
steps=[
S('Arrays',
"`np.array([...])` makes an array. `a * 10`, `a + b`, `a.mean()`, `a.max()` work on every element.\n\nScale the array so its largest value becomes 1.",
'''
import numpy as np
a = np.array([2.0, 4, 6, 8])
scaled = ___
print(scaled, a.mean())
''','''
import numpy as np
a = np.array([2.0, 4, 6, 8])
scaled = a / a.max()
print(scaled, a.mean())
''','''
assert np.allclose(scaled, [0.25, 0.5, 0.75, 1.0]), f"scaled should be a / a.max(), you have {scaled}"
''', "a / a.max()"),
S('Boolean masks are WHERE',
"`x > 0` gives an array of True/False. `x[x > 0]` keeps the rows where it's True. Summing True/False counts the Trues.",
'''
x = np.array([5, -3, 8, 0, -1, 7])
positives = ___
count_neg = (___).sum()
print(positives, count_neg)
''','''
x = np.array([5, -3, 8, 0, -1, 7])
positives = x[x > 0]
count_neg = (x < 0).sum()
print(positives, count_neg)
''','''
assert list(positives) == [5, 8, 7], "positives should be x[x > 0]"
assert count_neg == 2, "There are two negative values"
''', "positives = x[x > 0]; count_neg = (x < 0).sum()"),
S('Vectors: dot product and length',
"The dot product multiplies matching elements and sums them: `u @ v`. A vector's length is `np.linalg.norm(u)`. These two operations are everywhere in ML.",
'''
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
dot = ___
length_u = ___
print(dot, length_u)
''','''
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])
dot = u @ v
length_u = np.linalg.norm(u)
print(dot, length_u)
''','''
assert dot == 32, "1*4 + 2*5 + 3*6 = 32"
assert abs(length_u - 14 ** 0.5) < 1e-9, "length of u is sqrt(1 + 4 + 9)"
''', "dot = u @ v; length_u = np.linalg.norm(u)"),
S('Standardise a column',
"Standardising (z-scores) is the first thing done to most ML features: subtract the mean, divide by the standard deviation.",
'''
z = ___
print(z.round(3), z.mean().round(6), z.std())
''','''
z = (x - x.mean()) / x.std()
print(z.round(3), z.mean().round(6), z.std())
''','''
assert abs(z.mean()) < 1e-9 and abs(z.std() - 1) < 1e-9, "z should have mean 0 and standard deviation 1"
''', "(x - x.mean()) / x.std()"),
]),

dict(id='W4', title='Matrices: tables of numbers', packages=['numpy'], intro=
"A matrix is a table of numbers: rows are records, columns are features. Linear algebra in ML is mostly these few operations.",
steps=[
S('Shape and axis aggregates',
"`M.shape` is (rows, columns). `M.sum(axis=0)` sums down each column (like SUM per column); `axis=1` sums across each row.",
'''
import numpy as np
M = np.array([[1, 2, 3],
              [4, 5, 6]])
rows, cols = M.shape
col_sums = ___
row_sums = ___
print(rows, cols, col_sums, row_sums)
''','''
import numpy as np
M = np.array([[1, 2, 3],
              [4, 5, 6]])
rows, cols = M.shape
col_sums = M.sum(axis=0)
row_sums = M.sum(axis=1)
print(rows, cols, col_sums, row_sums)
''','''
assert (rows, cols) == (2, 3)
assert list(col_sums) == [5, 7, 9], "col_sums = M.sum(axis=0)"
assert list(row_sums) == [6, 15], "row_sums = M.sum(axis=1)"
''', "axis=0 goes down the columns"),
S('Matrix multiplication',
"`A @ B` multiplies matrices: each output cell is a dot product of a row of A with a column of B. `A.T` is the transpose.",
'''
A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 1], [1, 0]])
C = ___
print(C)
print(A.T)
''','''
A = np.array([[1, 2], [3, 4]])
B = np.array([[0, 1], [1, 0]])
C = A @ B
print(C)
print(A.T)
''','''
assert np.array_equal(C, [[2, 1], [4, 3]]), "C should be A @ B (this B swaps A's columns)"
''', "C = A @ B"),
S('Matrix times vector = weighted sum',
"`A @ w` gives each row's weighted sum, which is exactly what a linear model does to every record at once.",
'''
w = np.array([0.5, 0.5])
scores = ___
print(scores)
''','''
w = np.array([0.5, 0.5])
scores = A @ w
print(scores)
''','''
assert np.allclose(scores, [1.5, 3.5]), "scores = A @ w averages each row"
''', "scores = A @ w"),
S('Solving equations',
"`np.linalg.solve(A, b)` finds x with A x = b. Here: x + 2y = 5 and 3x + 4y = 11.",
'''
b = np.array([5, 11])
sol = ___
print(sol)
''','''
b = np.array([5, 11])
sol = np.linalg.solve(A, b)
print(sol)
''','''
assert np.allclose(sol, [1, 2]), "The solution is x=1, y=2"
''', "np.linalg.solve(A, b)"),
]),

dict(id='W5', title='pandas: the SQL you already know', packages=['numpy', 'pandas'], intro=
"pandas DataFrames are tables in Python. Every SQL clause you use daily has a one-line pandas equivalent.",
steps=[
S('Create a table and filter (WHERE)',
"`df[df['col'] > value]` keeps the matching rows, like WHERE.",
'''
import pandas as pd
customers = pd.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "city": ["HYD", "BLR", "HYD", "MUM", "BLR"],
    "balance": [52000, 18000, 75000, 30000, 9000],
})
print(customers)

rich = customers[___]
print(rich)
''','''
import pandas as pd
customers = pd.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "city": ["HYD", "BLR", "HYD", "MUM", "BLR"],
    "balance": [52000, 18000, 75000, 30000, 9000],
})
print(customers)

rich = customers[customers["balance"] > 50000]
print(rich)
''','''
assert list(rich["id"]) == [1, 3], "rich should be the customers with balance > 50000 (ids 1 and 3)"
''', 'customers["balance"] > 50000'),
S('GROUP BY',
"`df.groupby('col')['other'].mean()` is SELECT col, AVG(other) ... GROUP BY col.",
'''
avg_by_city = customers.groupby(___)[___].mean()
print(avg_by_city)
''','''
avg_by_city = customers.groupby("city")["balance"].mean()
print(avg_by_city)
''','''
assert avg_by_city["HYD"] == 63500 and avg_by_city["BLR"] == 13500, "Group by city, average balance"
''', 'groupby("city")["balance"]'),
S('JOIN',
"`left.merge(right, left_on=..., right_on=..., how='inner')` is an INNER JOIN.",
'''
txns = pd.DataFrame({"cust_id": [1, 3, 3, 4, 2], "amount": [500, 200, 300, 50, 100]})
merged = txns.merge(customers, left_on=___, right_on=___, how="inner")
spend_by_city = merged.groupby("city")["amount"].sum()
print(spend_by_city)
''','''
txns = pd.DataFrame({"cust_id": [1, 3, 3, 4, 2], "amount": [500, 200, 300, 50, 100]})
merged = txns.merge(customers, left_on="cust_id", right_on="id", how="inner")
spend_by_city = merged.groupby("city")["amount"].sum()
print(spend_by_city)
''','''
assert spend_by_city.to_dict() == {"BLR": 100, "HYD": 1000, "MUM": 50}, f"Expected BLR 100, HYD 1000, MUM 50; got {spend_by_city.to_dict()}"
''', 'left_on="cust_id", right_on="id"'),
S('ORDER BY ... LIMIT and computed columns',
"`sort_values(col, ascending=False).head(n)` is ORDER BY col DESC LIMIT n. `df['new'] = expression` adds a computed column.",
'''
top2 = customers.sort_values(___, ascending=False).head(___)
customers["balance_lakh"] = ___
print(top2)
print(customers)
''','''
top2 = customers.sort_values("balance", ascending=False).head(2)
customers["balance_lakh"] = customers["balance"] / 100000
print(top2)
print(customers)
''','''
assert list(top2["id"]) == [3, 1], "Top 2 by balance are ids 3 and 1"
assert abs(customers.loc[0, "balance_lakh"] - 0.52) < 1e-9, "balance_lakh = balance / 100000"
''', 'sort_values("balance", ascending=False).head(2)'),
]),

dict(id='W6', title='Plotting with matplotlib', packages=['numpy', 'matplotlib'], intro=
"You'll plot learning curves constantly. Three plot types cover most of it. Plots appear under the code when you press Run.",
steps=[
S('Line plot',
"`plt.plot(xs, ys)` draws a line. `np.linspace(0, 10, 100)` makes 100 evenly spaced points.",
'''
import numpy as np
import matplotlib.pyplot as plt
xs = np.linspace(0, 10, 100)
ys = ___
plt.plot(xs, ys)
plt.title("sin(x)")
''','''
import numpy as np
import matplotlib.pyplot as plt
xs = np.linspace(0, 10, 100)
ys = np.sin(xs)
plt.plot(xs, ys)
plt.title("sin(x)")
''','''
assert np.allclose(ys, np.sin(xs)), "ys should be np.sin(xs)"
assert plt.gcf().axes and plt.gcf().axes[0].lines, "Draw the line with plt.plot"
''', "ys = np.sin(xs)"),
S('Histogram',
"`plt.hist(data, bins=30)` shows a distribution. `np.random.default_rng(0)` makes a reproducible random generator.",
'''
rng = np.random.default_rng(0)
data = rng.normal(size=___)
plt.hist(data, bins=___)
''','''
rng = np.random.default_rng(0)
data = rng.normal(size=1000)
plt.hist(data, bins=30)
''','''
assert len(data) == 1000, "Draw 1000 samples"
assert len(plt.gca().patches) == 30, "Use 30 bins"
''', "size=1000, bins=30"),
S('Scatter with labels',
"Always label your axes; future-you will thank you.",
'''
x = rng.uniform(0, 10, 50)
y = 2 * x + rng.normal(0, 2, 50)
plt.scatter(x, y)
plt.xlabel(___)
plt.ylabel(___)
''','''
x = rng.uniform(0, 10, 50)
y = 2 * x + rng.normal(0, 2, 50)
plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
''','''
ax = plt.gca()
assert ax.get_xlabel() and ax.get_ylabel(), "Label both axes"
''', 'plt.xlabel("x")'),
]),

dict(id='W7', title='Simulation: let the computer flip coins', packages=['numpy', 'matplotlib'], intro=
"When maths gets hard, simulate. This is how you'll check every probability result in the next courses.",
steps=[
S('Flip 1,000 coins',
"`rng.integers(0, 2, size=n)` gives n random 0s and 1s. The mean is the fraction of heads.",
'''
import numpy as np
rng = np.random.default_rng(42)
flips = rng.integers(0, 2, size=___)
frac_heads = ___
print(frac_heads)
''','''
import numpy as np
rng = np.random.default_rng(42)
flips = rng.integers(0, 2, size=1000)
frac_heads = flips.mean()
print(frac_heads)
''','''
assert len(flips) == 1000
assert abs(frac_heads - flips.mean()) < 1e-12 and 0.4 < frac_heads < 0.6, "frac_heads = flips.mean()"
''', "flips.mean()"),
S('Two dice: P(sum = 7)',
"`size=(100000, 2)` gives 100,000 rows of 2 dice. Sum each row, compare with 7, take the mean.",
'''
rolls = rng.integers(1, 7, size=(100000, 2))
p7 = (___ == 7).mean()
print(p7, 1 / 6)
''','''
rolls = rng.integers(1, 7, size=(100000, 2))
p7 = (rolls.sum(axis=1) == 7).mean()
print(p7, 1 / 6)
''','''
assert abs(p7 - 1 / 6) < 0.01, f"p7 should be close to 1/6 = 0.1667, you have {p7}"
''', "rolls.sum(axis=1)"),
S('Watch the average settle',
"`np.cumsum` gives running totals. Divide by 1, 2, 3, ... to get the running average. This is the law of large numbers.",
'''
import matplotlib.pyplot as plt
running = np.cumsum(flips) / ___
plt.plot(running)
plt.axhline(0.5, color="gray")
''','''
import matplotlib.pyplot as plt
running = np.cumsum(flips) / np.arange(1, len(flips) + 1)
plt.plot(running)
plt.axhline(0.5, color="gray")
''','''
assert len(running) == 1000 and abs(running[-1] - frac_heads) < 1e-12, "Divide by np.arange(1, len(flips) + 1)"
''', "np.arange(1, len(flips) + 1)"),
S('A reusable simulator',
"Write `simulate(p, n)` that returns the fraction of successes in n trials with success probability p. Hint: `rng.random(n) < p`.",
'''
def simulate(p, n):
    rng = np.random.default_rng(0)
    return ___

print(simulate(0.3, 100000))
''','''
def simulate(p, n):
    rng = np.random.default_rng(0)
    return (rng.random(n) < p).mean()

print(simulate(0.3, 100000))
''','''
assert abs(simulate(0.3, 100000) - 0.3) < 0.01 and abs(simulate(0.9, 100000) - 0.9) < 0.01, "simulate(p, n) should be close to p"
''', "(rng.random(n) < p).mean()"),
]),

dict(id='W8', title='Your first learning loop: gradient descent', packages=['numpy', 'matplotlib'], intro=
"Every model in these courses, from linear regression to DQN, learns the same way: define a loss, compute its gradient, step downhill, repeat. You'll write that loop now in about 15 lines.",
steps=[
S('Data and a loss',
"We make data from y = 3x + 2 plus noise. The loss measures how wrong a guess (w, b) is: the mean squared error.",
'''
import numpy as np
rng = np.random.default_rng(1)
x = rng.uniform(0, 10, 100)
y = 3 * x + 2 + rng.normal(0, 1, 100)

def mse(w, b):
    pred = w * x + b
    return ___

print(mse(0, 0), mse(3, 2))
''','''
import numpy as np
rng = np.random.default_rng(1)
x = rng.uniform(0, 10, 100)
y = 3 * x + 2 + rng.normal(0, 1, 100)

def mse(w, b):
    pred = w * x + b
    return np.mean((pred - y) ** 2)

print(mse(0, 0), mse(3, 2))
''','''
assert abs(mse(3, 2) - np.mean((3 * x + 2 - y) ** 2)) < 1e-9, "mse = mean of (pred - y) squared"
assert mse(0, 0) > 100 > mse(3, 2), "A bad guess should have a much larger loss"
''', "np.mean((pred - y) ** 2)"),
S('The gradient',
"The gradient tells you which way is uphill. For MSE: dw = mean(2 (pred - y) x) and db = mean(2 (pred - y)). The check compares your formula with a numerical estimate.",
'''
def grads(w, b):
    err = w * x + b - y
    dw = ___
    db = ___
    return dw, db

print(grads(0, 0))
''','''
def grads(w, b):
    err = w * x + b - y
    dw = np.mean(2 * err * x)
    db = np.mean(2 * err)
    return dw, db

print(grads(0, 0))
''','''
h = 1e-5
for w0, b0 in [(0, 0), (1.5, -1)]:
    dw, db = grads(w0, b0)
    nw = (mse(w0 + h, b0) - mse(w0 - h, b0)) / (2 * h)
    nb = (mse(w0, b0 + h) - mse(w0, b0 - h)) / (2 * h)
    assert abs(dw - nw) < 1e-3 * max(1, abs(nw)), f"dw is off: you have {dw:.3f}, numerically it's {nw:.3f}"
    assert abs(db - nb) < 1e-3 * max(1, abs(nb)), f"db is off: you have {db:.3f}, numerically it's {nb:.3f}"
''', "dw = np.mean(2 * err * x); db = np.mean(2 * err)"),
S('The learning loop',
"Start at w = b = 0 and repeatedly step against the gradient: `w -= lr * dw`. Keep the loss at each step.",
'''
w, b, lr = 0.0, 0.0, 0.01
losses = []
for step in range(3000):
    dw, db = grads(w, b)
    w = ___
    b = ___
    losses.append(mse(w, b))
print(w, b)
''','''
w, b, lr = 0.0, 0.0, 0.01
losses = []
for step in range(3000):
    dw, db = grads(w, b)
    w = w - lr * dw
    b = b - lr * db
    losses.append(mse(w, b))
print(w, b)
''','''
assert abs(w - 3) < 0.2 and abs(b - 2) < 0.6, f"Expected w near 3 and b near 2, you have w={w:.2f}, b={b:.2f}"
''', "w = w - lr * dw"),
S('Plot the learning curve',
"Plot `losses`. A falling curve is what learning looks like. You'll see this plot in every course from here on.",
'''
import matplotlib.pyplot as plt
plt.plot(___)
plt.yscale("log")
plt.xlabel("step")
plt.ylabel("loss")
''','''
import matplotlib.pyplot as plt
plt.plot(losses)
plt.yscale("log")
plt.xlabel("step")
plt.ylabel("loss")
''','''
assert plt.gca().lines and len(plt.gca().lines[0].get_ydata()) == 3000, "Plot the losses list"
''', "plt.plot(losses)"),
]),
]
