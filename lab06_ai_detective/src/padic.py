"""Конечная p-adic модель дерева глубины 3; равные пути имеют расстояние 0."""
from itertools import groupby

def code(path):
    if len(path)!=3 or any(x not in (0,1,2) for x in path):
        raise ValueError('Expected three ternary digits')
    return sum(x*3**i for i,x in enumerate(path))

def distance(a,b):
    code(a);code(b)
    for k,(x,y) in enumerate(zip(a,b)):
        if x!=y:return 3.0**(-k)
    return 0.0

def branches(claims):
    ordered=sorted(claims,key=lambda c:(c['path'],c['event_time'],c['claim_id']))
    return [(list(path),list(items)) for path,items in groupby(ordered,key=lambda c:tuple(c['path']))]
