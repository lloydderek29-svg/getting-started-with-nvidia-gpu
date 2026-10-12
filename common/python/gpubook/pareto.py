
def pareto_front(points):
    """points: (name, cost, quality). Lower cost and higher
    quality are better. Return names not dominated."""
    keep = []
    for name, cost, quality in points:
        dominated = any(
            c2 <= cost and q2 >= quality
            and (c2 < cost or q2 > quality)
            for n2, c2, q2 in points if n2 != name)
        if not dominated:
            keep.append(name)
    return keep
# chapters/ch08_generative/tests/test_pareto.py
from gpubook.pareto import pareto_front


def test_dominated_points_are_removed():
    pts = [("a", 10.0, 0.5), ("b", 8.0, 0.5),
           ("c", 12.0, 0.6), ("d", 20.0, 0.55)]
    assert set(pareto_front(pts)) == {"b", "c"}
