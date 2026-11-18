import json, re
from content import *

raw = {c['id']: c for c in json.load(open('raw.json'))}

def topic_of(name):
    t = re.sub(r'^\s*week[\s\-]*\d+\s*[:\-·]?\s*', '', name, flags=re.I).strip()
    return t

def clean_lec(code, t):
    t = t.replace('\t', ' ').strip()
    m = re.match(r'noc21-cs104-lec(\d+)', t)
    if m: return f'Lecture {int(m.group(1)):02d}'
    return t

def week_no(title):
    m = re.search(r'Week_?(\d+)', title)
    return int(m.group(1)) if m else None

out = {'courses': {}, 'books': BOOKS, 'phases_full': PHASES_FULL, 'phases_core': PHASES_CORE}
for code, (nid, short, est, hue) in COURSES.items():
    c = raw[nid]
    units = c['u']
    weeks = []
    if code == 'OC':
        lecs = [(clean_lec(code, l[0]), l[1], u[0]) for u in units for l in u[1]]
        for i in range(0, len(lecs), 5):
            chunk = lecs[i:i+5]
            topics = []
            for _, _, un in chunk:
                un = un.replace("\\'", "'").strip()
                if un not in topics: topics.append(un)
            weeks.append({'topic': '; '.join(topics), 'lectures': [[a, b] for a, b, _ in chunk]})
    else:
        for u in units:
            topic = topic_of(u[0])
            if code == 'PCS' or re.fullmatch(r'week\s*\d+', u[0].strip(), re.I) or topic == '':
                topic = ''
            if u[0].startswith('Live Session'): topic = 'Live problem-solving session (optional)'
            weeks.append({'topic': topic, 'lectures': [[clean_lec(code, l[0]), l[1]] for l in u[1]]})
    # assignments
    amap = {}
    res = []
    for title, url in c['a']:
        n = week_no(title)
        if n is None:
            res.append([title, url]); continue
        n = max(n, 1)
        n0 = week_no(title)
        amap.setdefault(n, []).append(['NPTEL assignment, week %d%s' % (n0, ' (warm-up)' if n0 == 0 else ''), url])
    for i, w in enumerate(weeks, 1):
        w['n'] = i
        w['assign'] = amap.get(i, [])
        if code == 'RL':
            w['assign'] = [['NPTEL assignment %d with solution (Prof. Ravindran\'s course page)' % i, RL_ASSIGN_URL]] if i <= 11 else []
        t = TASKS.get(code)
        w['task'] = t[i-1] if t and i-1 < len(t) else None
        r = READING.get(code, {}).get(i)
        w['reading'] = {'book': BOOKS[r[0]][0], 'ref': r[1], 'url': BOOKS[r[0]][1]} if r else None
    if code == 'MC':
        weeks[-1]['task'] = TASKS['MC'][8]
    out['courses'][code] = {
        'code': code, 'id': nid, 'name': short, 'title': c['t'], 'prof': c['p'].replace('\r', '').replace('\n', ' '),
        'inst': c['i'], 'est': est, 'hue': hue, 'url': 'https://nptel.ac.in/courses/' + nid,
        'resources': res, 'weeks': weeks, 'finals': FINALS[code],
    }

out['projects'] = [dict(id=p[0], anchor=p[1], hours=p[2], core=p[3], title=p[4], desc=p[5], deliverable=p[6]) for p in PROJECTS]
out['phase7'] = [dict(id=p[0], kind=p[1], size=p[2], core=p[3], title=p[4], desc=p[5], url=p[6]) for p in PHASE7]

import labs_warmup, labs_p1, labs_iml, labs_dl, labs_rl, labs_ba, labs_lambda
from nb_specs import NOTEBOOKS
LABS = labs_warmup.WARMUP + labs_lambda.LAMBDA + labs_p1.P1 + labs_p1.MC + labs_iml.IML + labs_iml.PROJ + labs_dl.DL_ALL + labs_rl.RL + labs_ba.BA_ALL
out['labs'] = {l['id']: l for l in LABS}
out['warmup'] = ['W1', 'W2', 'WL1', 'W3', 'W4', 'W5', 'WL2', 'W6', 'W7', 'W8']
labmap = {}
for l in LABS:
    m = re.fullmatch(r'(MC|IML|DL|BA)(\d+)', l['id'])
    if m:
        labmap[f'{m.group(1)}-w{m.group(2)}-t'] = l['id']
    elif re.fullmatch(r'P\d+', l['id']):
        labmap[l['id']] = l['id']
COLAB = 'https://colab.research.google.com/github/prdurgam-edu/rl-study-desk/blob/main/notebooks/'
out['colab'] = {nb['id'].replace('NB_', ''): {'url': COLAB + nb['file'], 'title': nb['title']} for nb in NOTEBOOKS}
out['labmap'] = labmap
json.dump(out, open('curriculum.json', 'w'), ensure_ascii=False, separators=(',', ':'))
tot = sum(len(w['lectures']) for c in out['courses'].values() for w in c['weeks'])
print('lectures', tot, 'bytes', len(json.dumps(out, separators=(',', ':'))))
for code, c in out['courses'].items():
    print(code, len(c['weeks']), [len(w['lectures']) for w in c['weeks']], sum(1 for w in c['weeks'] if w['assign']), sum(1 for w in c['weeks'] if w['task']), sum(1 for w in c['weeks'] if w['reading']))
print(out['courses']['OC']['weeks'][0]['topic'])
print(out['courses']['MC']['weeks'][8])
print(out['courses']['PCS']['weeks'][0]['assign'])
