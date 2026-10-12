from gpubook.llm.perplexity import window_plan


def test_windows_for_small_example():
    plan = list(window_plan(10, 4, 2))
    assert plan == [(0, 4, 4), (2, 6, 2), (4, 8, 2), (6, 10, 2)]


def test_every_token_is_scored_once():
    for n in (1, 7, 100, 2047, 5000):
        total = sum(new for _, _, new in window_plan(n, 64, 32))
        assert total == n
