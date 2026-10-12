from gpubook.llm.paging import contiguous_waste, paged_waste


def test_contiguous_reservation_wastes_most_of_it():
    assert contiguous_waste([100, 300], 2048) == 2048 * 2 - 400


def test_paged_waste_is_below_one_block_per_sequence():
    lens = [100, 300, 17, 16]
    waste = paged_waste(lens, 16)
    assert waste == 12 + 4 + 15 + 0
    assert waste < 16 * len(lens)
