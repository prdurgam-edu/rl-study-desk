import os, sys, time, importlib, traceback
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mods = sys.argv[1:] or ['labs_warmup']
fails = 0
for m in mods:
    mod = importlib.import_module(m)
    labs = [v for k, v in vars(mod).items() if k.isupper() and isinstance(v, list)][0]
    for lab in labs:
        ns = {}
        t0 = time.time()
        for i, st in enumerate(lab['steps']):
            if '___' not in st['starter'] and not st.get('free'):
                print(f"  note {lab['id']} step {i+1}: starter has no blanks")
            try:
                exec(compile(st['solution'], f"{lab['id']}-{i+1}-sol", 'exec'), ns)
                exec(compile(st['check'], f"{lab['id']}-{i+1}-chk", 'exec'), ns)
            except Exception:
                fails += 1
                print(f"FAIL {lab['id']} step {i+1}: {st['title']}")
                traceback.print_exc(limit=2)
            plt.close('all')
        print(f"ok {lab['id']} ({len(lab['steps'])} steps, {time.time()-t0:.1f}s)")
print('FAILURES', fails)
