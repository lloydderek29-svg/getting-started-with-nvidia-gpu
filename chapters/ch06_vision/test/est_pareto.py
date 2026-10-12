from accuracy_latency import pareto_front


def test_dominated_points_are_removed():
    pts = [("a", 10.0, 0.50), ("b", 8.0, 0.50),
           ("c", 12.0, 0.60), ("d", 20.0, 0.55)]
    assert set(pareto_front(pts)) == {"b", "c"}
