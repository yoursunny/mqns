from collections import defaultdict

import pytest

from mqns.models.core import TrafficMatrix


def test_bad_input():
    with pytest.raises(ValueError, match="should have \\(3,3\\) shape"):
        TrafficMatrix([[0, 1, 1]], 3)

    with pytest.raises(ValueError, match="should have \\(3,3\\) shape"):
        TrafficMatrix([[0, 1, 1], [1, 0]], 3)

    with pytest.raises(ValueError, match="must be nonnegative"):
        TrafficMatrix([[0, -1, 1], [0, 0, 0], [0, 0, 0]], 3)

    with pytest.raises(ValueError, match="diagonal elements .* must be exactly zero"):
        TrafficMatrix([[1, 1, 1]] * 3, 3)

    with pytest.raises(ValueError, match="input is zero"):
        TrafficMatrix([[0, 0, 0]] * 3, 3)


def test_sample1():
    tm = TrafficMatrix([[0, 2, 0], [0, 0, 0], [0, 0, 0]], 3)
    assert tm[0, 1] == pytest.approx(1.0, abs=1e-6)
    assert tm.sample() == (0, 1)


def test_sample2():
    tm = TrafficMatrix([[0, 3, 0], [0, 0, 1], [0, 0, 0]], 3)
    assert tm[0, 1] == pytest.approx(3 / 4, abs=1e-6)
    assert tm[1, 2] == pytest.approx(1 / 4, abs=1e-6)

    samples = defaultdict[tuple[int, int], int](lambda: 0)
    for _ in range(100):
        samples[tm.sample()] += 1

    assert len(samples) == 2
    assert samples.get((0, 1)) == pytest.approx(75, abs=20)
