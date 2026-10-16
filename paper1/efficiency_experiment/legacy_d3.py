"""Self-contained adapter of the original fixed 48-cell linear-Gaussian D3 panel.

The graph construction, insertion order, topological order and floating-point
simulation operations are retained from ``paper1.bench.designs`` (itself ported
from ``charmv2/第三批/code/b3/b3_gen_d3.py``).  This small copy deliberately has
no dependency on discovery, certification or the main paper configuration, so
the generator and efficiency experiment can be exported on their own.

``cells`` fixes the original factorial panel; ``make_design`` freezes one model;
``simulate`` consumes a caller-owned numpy Generator and draws independent
N(0,1) errors.  It does not implement the newer configurable random-DAG generator.
NIE is the path through the target M; NDE also contains the optional parallel M2
path.  All effects are for a unit exposure contrast.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

A_XM, B_MY, D_XY = 0.6, 0.5, 0.2
A_XM2, B_M2Y = 0.5, 0.4


def make_design(base='split', n_inst=0, n_pm=0, n_py=0, n_weak=0, m2=True, dpost=True, inst=0.7, pred=0.5, weak=0.15):
    E = []
    if base == 'split':
        E += [('C1', 'X', .5), ('C1', 'M', .5), ('C2', 'X', .5), ('C2', 'Y', .5), ('C3', 'M', .5), ('C3', 'Y', .5)]
    elif base == 'common':
        for c in ('C1', 'C2', 'C3'):
            E += [(c, 'X', .4), (c, 'M', .4), (c, 'Y', .4)]
    else:
        raise ValueError(base)
    E += [('C1', 'C2', .4), ('X', 'M', A_XM), ('M', 'Y', B_MY), ('X', 'Y', D_XY)]
    roles = {c: 'baseline' for c in ('C1', 'C2', 'C3')}
    for k in range(n_inst):
        E.append((f'I{k + 1}', 'X', inst)); roles[f'I{k + 1}'] = 'instrument'
    for k in range(n_pm):
        E.append((f'PM{k + 1}', 'M', pred)); roles[f'PM{k + 1}'] = 'pure_M_predictor'
    for k in range(n_py):
        E.append((f'PY{k + 1}', 'Y', pred)); roles[f'PY{k + 1}'] = 'pure_Y_predictor'
    if n_weak:
        E += [('WM', 'M', weak), ('WY', 'Y', weak)]; roles['WM'] = 'weak_M_parent'; roles['WY'] = 'weak_Y_parent'
    if m2:
        E += [('X', 'M2', A_XM2), ('M2', 'Y', B_M2Y), ('C2', 'M2', .4)]; roles['M2'] = 'parallel_mediator'
    if dpost:
        E += [('X', 'D', .5), ('C1', 'D', .4)]; roles['D'] = 'post_exposure'
    G = nx.DiGraph(); G.add_weighted_edges_from(E)
    order = list(nx.topological_sort(G))
    truth = dict(NIE=A_XM * B_MY, NDE=D_XY + (A_XM2 * B_M2Y if m2 else 0.0))
    pool = sorted(set(G.nodes) - nx.descendants(G, 'X') - {'X', 'M', 'Y'})
    return dict(G=G, order=order, roles=roles, truth=truth, pool=pool,
                factors=dict(base=base, n_inst=n_inst, n_pm=n_pm, n_py=n_py, n_weak=n_weak, m2=m2, dpost=dpost))


def simulate(design, n, rng):
    """returns (V, cols): V is (n, p) in the column order cols."""
    G, order = design['G'], design['order']
    ix = {v: i for i, v in enumerate(order)}
    V = rng.normal(size=(n, len(order)))
    for v in order:
        for u in G.predecessors(v):
            V[:, ix[v]] += G[u][v]['weight'] * V[:, ix[u]]
    return V, list(order)


def cells():
    out = []
    for base in ('split', 'common'):
        for n_inst in (0, 1, 3):
            for n_pm in (0, 2):
                for n_py in (0, 2):
                    for n_weak in (0, 2):
                        out.append(dict(base=base, n_inst=n_inst, n_pm=n_pm, n_py=n_py, n_weak=n_weak))
    return out

