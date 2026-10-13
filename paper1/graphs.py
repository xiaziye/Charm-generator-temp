"""Graph criteria on DAGs (networkx DiGraph).

Ported unchanged from the frozen prototype (charmv2/第三批/code/b1/eif_core.py and b2/b2_common.py):
d-separation, proper causal nodes, forbidden sets, the adjustment criterion (Shpitser et al. 2010; Perkovic et al.
2018), the SV2011 mediation criterion, stage-specific pair admissibility, and the O-sets of Henckel, Perkovic &
Maathuis (2022).  Plus two helpers for the bench: intermediate confounders and the non-descendant pool.
"""
from __future__ import annotations

import networkx as nx


def d_sep(G, X, Y, Z):
    X, Y, Z = set(X), set(Y), set(Z)
    if not X or not Y:
        return True
    try:
        return nx.is_d_separator(G, X, Y, Z)
    except AttributeError:  # networkx < 3.3
        return nx.d_separated(G, X, Y, Z)


def _reach_avoiding(G, sources, avoid):
    seen, stack = set(), list(sources)
    while stack:
        u = stack.pop()
        for w in G.successors(u):
            if w in avoid or w in seen:
                continue
            seen.add(w)
            stack.append(w)
    return seen


def proper_causal_nodes(G, X, Y):
    X, Y = set(X), set(Y)
    fwd = _reach_avoiding(G, X, X)
    H = G.copy()
    H.remove_nodes_from(X)
    anc = set()
    for y in Y:
        if y in H:
            anc |= nx.ancestors(H, y) | {y}
    return fwd & anc


def forbidden(G, X, Y):
    out = set()
    for w in proper_causal_nodes(G, X, Y):
        out |= nx.descendants(G, w) | {w}
    return out


def proper_backdoor_graph(G, X, Y):
    X = set(X)
    cn = proper_causal_nodes(G, X, Y)
    H = G.copy()
    for x in X:
        for w in list(G.successors(x)):
            if w in cn:
                H.remove_edge(x, w)
    return H


def adj_valid(G, X, Y, C):
    """Adjustment criterion (Shpitser et al. 2010; Perkovic et al. 2018) in a DAG."""
    X, Y, C = set(X), set(Y), set(C)
    if C & (X | Y):
        return False
    if C & forbidden(G, X, Y):
        return False
    return d_sep(proper_backdoor_graph(G, X, Y), X, Y, C)


def mediation_valid(G, A, M, Y, C):
    """SV2011 Theorem 2: mediation formula valid iff C valid for ({A,M},Y) and for (A,M)."""
    return adj_valid(G, {A, M}, {Y}, C) and adj_valid(G, {A}, {M}, C)


def pair_admissible(G, A, M, Y, CM, CY):
    """Stage-specific pair: union valid (SV2011) + M _||_ CY\\CM | A,CM  + Y _||_ CM\\CY | A,M,CY."""
    CM, CY = set(CM), set(CY)
    U = CM | CY
    if not mediation_valid(G, A, M, Y, U):
        return False
    if not d_sep(G, {M}, CY - CM, {A} | CM):
        return False
    if not d_sep(G, {Y}, CM - CY, {A, M} | CY):
        return False
    return True


def pair_admissible_set(G, A, Ms, Y, CM, CY):
    """Stage-specific admissibility for a mediator SET (SV2011 with M a set + the two d-separations)."""
    Ms, CM, CY = set(Ms), set(CM), set(CY)
    U = CM | CY
    if not (adj_valid(G, {A} | Ms, {Y}, U) and adj_valid(G, {A}, Ms, U)):
        return False
    if not d_sep(G, Ms, CY - CM, {A} | CM):
        return False
    if not d_sep(G, {Y}, CM - CY, {A} | Ms | CY):
        return False
    return True


def o_set(G, X, Y):
    """O(X, Y, G) = pa(cn(X, Y, G)) \\ forb(X, Y, G), forb = de(cn) u X  (Henckel et al. 2022, Def. 3.12)."""
    X, Y = set(X), set(Y)
    cn = proper_causal_nodes(G, X, Y)
    pa = set()
    for w in cn:
        pa |= set(G.predecessors(w))
    return pa - forbidden(G, X, Y) - X


def graph_pair(G, A, Ms, Y):
    """(O(A, M), O({A} u M, Y)) for a mediator set Ms (tuple of node names)."""
    Ms = set(Ms)
    return tuple(sorted(o_set(G, {A}, Ms))), tuple(sorted(o_set(G, {A} | Ms, {Y})))


def oriented_supergraph(D, T, nodes=None):
    """True iff every edge of T (restricted to `nodes`) is present in D with the same orientation."""
    for u, v in T.edges():
        if nodes is not None and (u not in nodes or v not in nodes):
            continue
        if not D.has_edge(u, v):
            return False
    return True


def pool_of(dags, X, M, Y):
    """Columns that are non-descendants of X in EVERY DAG of the family, minus X, M, Y (empty family -> None)."""
    pool = None
    for D in dags:
        nd = set(D.nodes) - nx.descendants(D, X) - {X, M, Y}
        pool = nd if pool is None else pool & nd
    return pool


def intermediate_confounders(G, X, M, Y):
    """Descendants of X that are ancestors of M and reach Y avoiding M (a 'recanting witness'): with one of them in
    the true graph, no adjustment pair identifies the natural effects through M."""
    H = G.copy(); H.remove_node(M)
    return sorted(w for w in (nx.descendants(G, X) & nx.ancestors(G, M)) - {M}
                  if w in H and Y in nx.descendants(H, w))
