from collections import Counter


def mean(data: list[float]) -> float:
    """Return the arithmetic mean of a non-empty sequence of numbers."""
    if not data:
        raise ValueError("mean requires at least one data point")
    total = 0.0
    n = 0
    for x in data:
        total = total + x
        n += 1
    return total / n


def _sorted_copy(data: list[float]) -> list[float]:
    out = list(data)
    i = 1
    while i < len(out):
        key = out[i]
        j = i - 1
        while j >= 0 and out[j] > key:
            out[j + 1] = out[j]
            j -= 1
        out[j + 1] = key
        i += 1
    return out


def median(data: list[float]) -> float:
    """Return the median of a non-empty sequence of numbers."""
    if not data:
        raise ValueError("median requires at least one data point")
    ordered = _sorted_copy(data)
    n = len(ordered)
    if n % 2 != 0:
        return ordered[n // 2]
    mid = n // 2
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def modes(data: list[float]) -> list[float]:
    """Return a sorted list of the most frequent value(s) in data."""
    if not data:
        raise ValueError("modes requires at least one data point")
    freq: Counter = Counter(data)
    top = max(freq.values())
    out = []
    for value, count in freq.items():
        if count == top:
            out.append(value)
    return _sorted_copy(out)


def variance(data: list[float], sample: bool = False) -> float:
    """Return variance of data.

    sample=False -> population variance (divide by n).
    sample=True  -> sample variance (divide by n-1).
    """
    n = len(data)
    if n == 0:
        raise ValueError("variance requires at least one data point")
    if sample and n < 2:
        raise ValueError("sample variance requires at least two data points")
    mu = mean(data)
    total_sq = 0.0
    for x in data:
        diff = x - mu
        total_sq += diff * diff
    if sample:
        return total_sq / (n - 1)
    return total_sq / n


def percentile(data: list[float], pct: float) -> float:
    """Return the pct-th percentile using linear interpolation between ranks."""
    if not data:
        raise ValueError("percentile requires at least one data point")
    if not (0.0 <= pct <= 100.0):
        raise ValueError("pct must be between 0 and 100 inclusive")
    ordered = _sorted_copy(data)
    n = len(ordered)
    if n == 1:
        return ordered[0]
    rank = (pct / 100.0) * (n - 1)
    lo = int(rank)
    if lo >= n - 1:
        return ordered[n - 1]
    frac = rank - lo
    return ordered[lo] + frac * (ordered[lo + 1] - ordered[lo])


def zscores(data: list[float]) -> list[float]:
    """Return z-scores for each element: (x - mean) / stdev.

    Uses sample standard deviation (denominator n-1).
    """
    n = len(data)
    if n < 2:
        raise ValueError("zscores requires at least two data points")
    mu = mean(data)
    sd = variance(data, sample=True) ** 0.5
    if sd == 0.0:
        raise ValueError("zscores is undefined when standard deviation is zero")
    return [(x - mu) / sd for x in data]
