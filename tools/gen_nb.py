"""Generate guided Colab notebooks and test them by running the solutions and checks."""
import json, os, sys, time, re
from nb_specs import NOTEBOOKS, COMMON_HOW

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}

def code(src, hidden=False):
    meta = {"cellView": "form"} if hidden else {}
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [], "source": src.splitlines(True)}

def build(nb):
    cells = [md(f"# {nb['title']}\n\n{nb['intro']}\n\n{COMMON_HOW}"), md("## Setup\nRun this once."), code(nb['setup'].strip())]
    for i, st in enumerate(nb['steps'], 1):
        cells.append(md(f"## Step {i}: {st['title']}\n\n{st['text']}"))
        cells.append(code(st['starter']))
        cells.append(code("# Check: run me after the cell above\n" + st['check']))
        cells.append(code("#@title Solution (click ▸ to show)\n" + st['solution'], hidden=True))
    return {"cells": cells, "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 0}

def run(nb, full=False):
    os.environ["NB_TEST"] = "0" if full else "1"
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = {}
    strip = lambda s: "\n".join(l for l in s.splitlines() if not l.lstrip().startswith("!"))
    exec(strip(nb['setup']), ns)
    for i, st in enumerate(nb['steps'], 1):
        t = time.time()
        exec(strip(st['solution']), ns); exec(st['check'], ns); plt.close('all')
        print(f"  {nb['id']} step {i} ok ({time.time() - t:.1f}s)")

if __name__ == "__main__":
    os.makedirs("notebooks", exist_ok=True)
    for nb in NOTEBOOKS:
        json.dump(build(nb), open(os.path.join("notebooks", nb['file']), "w"), indent=1, ensure_ascii=False)
    which = sys.argv[1:]
    for nb in NOTEBOOKS:
        if which and nb['id'] not in which and 'all' not in which:
            continue
        print("testing", nb['id']); run(nb, full='--full' in sys.argv)
