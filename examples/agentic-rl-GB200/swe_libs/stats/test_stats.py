import pytest
from stats import mean, median, modes, percentile, variance, zscores


def test_mean_basic():
    assert mean([1, 2, 3, 4, 5]) == 3.0


def test_mean_single_value():
    assert mean([7]) == 7.0


def test_mean_negative_values():
    # (-4 + -2 + 0) / 3 = -2.0
    assert mean([-4, -2, 0]) == -2.0


def test_mean_empty_raises():
    with pytest.raises(ValueError):
        mean([])


def test_median_odd_count():
    assert median([9, 3, 7]) == 7.0


def test_median_even_count():
    # (4 + 6) / 2 = 5.0
    assert median([4, 8, 1, 6]) == 5.0


def test_median_single_value():
    assert median([42]) == 42.0


def test_median_does_not_mutate_input():
    data = [3, 1, 2]
    median(data)
    assert data == [3, 1, 2]


def test_modes_single():
    assert modes([1, 2, 2, 3]) == [2]


def test_modes_multiple_tie():
    # 1 appears twice, 2 appears twice -> both are modes
    assert modes([1, 2, 1, 2, 3]) == [1, 2]


def test_modes_empty_raises():
    with pytest.raises(ValueError):
        modes([])


def test_variance_population_basic():
    # mean=3, deviations [-2,-1,0,1,2] -> sum sq = 10 -> 10/5 = 2.0
    assert variance([1, 2, 3, 4, 5]) == 2.0


def test_variance_sample_basic():
    # same data, denominator n-1 = 4 -> 2.5
    assert variance([1, 2, 3, 4, 5], sample=True) == 2.5


def test_variance_constant_data_zero():
    assert variance([5, 5, 5]) == 0.0


def test_variance_sample_single_point_raises():
    with pytest.raises(ValueError):
        variance([1], sample=True)


def test_percentile_min_is_zeroth():
    assert percentile([1, 2, 3, 4], 0.0) == 1.0


def test_percentile_max_is_last():
    assert percentile([1, 2, 3, 4], 100.0) == 4.0


def test_percentile_median_like():
    # rank = 0.5 * 3 = 1.5 -> between ordered[1]=2 and ordered[2]=3 -> 2.5
    assert percentile([4, 1, 3, 2], 50.0) == 2.5


def test_percentile_mid_quarter():
    # n=5, pct=25 -> rank = 0.25*4 = 1.0 -> exactly ordered[1] = 2
    assert percentile([5, 1, 4, 2, 3], 25.0) == 2.0


def test_percentile_out_of_range_raises():
    with pytest.raises(ValueError):
        percentile([1, 2, 3], 101)


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50.0)


def test_zscores_basic():
    # mean=3, sample stdev = sqrt(2.5) = 1.5811...
    # z-scores: -2/1.5811 = -1.26491..., etc.
    z = zscores([1, 3, 5])
    assert z == [
        pytest.approx(-1.2649110640673518),
        pytest.approx(0.0),
        pytest.approx(1.2649110640673518),
    ]


def test_zscores_constant_data_raises():
    with pytest.raises(ValueError):
        zscores([7, 7, 7])


def test_zscores_single_point_raises():
    with pytest.raises(ValueError):
        zscores([1])
