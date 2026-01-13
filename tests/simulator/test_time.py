import pytest

from mqns.simulator import Time


def test_time_compare():
    t1 = Time(1, accuracy=1000000)
    t2 = Time.from_sec(1.1, accuracy=1000000)
    t3 = Time(0, accuracy=1000000)
    t4 = Time(1100000, accuracy=1000000)

    assert t1 == t1  # noqa: PLR0124
    assert t2 >= t1
    assert t1 <= t2
    assert t1 < t2
    assert t3 < t1
    assert t1 != t4
    assert t2 == t4

    t0 = Time.from_sec(1.1, accuracy=2000)
    assert t2 != t0
    assert pytest.raises(AssertionError, lambda: t3 < t0)

    assert t2 != 1
    assert t2 != "A"


def test_time_accuracy():
    t0 = Time.from_sec(1.0, accuracy=1000000)
    t1 = Time.from_sec(1.0, accuracy=2000)
    t2 = Time.from_sec(1.0, accuracy=3000)
    t3 = Time.from_sec(1.0, accuracy=1000000)
    t4 = Time.from_sec(1.0, accuracy=4000)

    assert t0.sec == pytest.approx(1.0)
    assert t1.sec == pytest.approx(1.0)
    assert t2.sec == pytest.approx(1.0)
    assert t3.sec == pytest.approx(1.0)
    assert t4.sec == pytest.approx(1.0)

    assert t0.accuracy == t3.accuracy
    assert t1.accuracy == 2000
    assert t2.accuracy == 3000
    assert t4.accuracy == 4000


def test_time_add_sub():
    t5 = Time.from_sec(5, accuracy=1000)

    t6a = t5 + 1
    t6b = t5 + 1.0
    t6c = t5 + Time.from_sec(1.0, accuracy=1000)
    assert t6a.sec == pytest.approx(6.0)
    assert t6b.sec == pytest.approx(6.0)
    assert t6c.sec == pytest.approx(6.0)

    assert pytest.raises(AssertionError, lambda: t5 + Time.from_sec(1.0, accuracy=2000))

    t3a = t5 - 2
    t3b = t5 - 2.0
    t3c = t5 - Time.from_sec(2.0, accuracy=1000)
    assert t3a.sec == pytest.approx(3.0)
    assert t3b.sec == pytest.approx(3.0)
    assert t3c.sec == pytest.approx(3.0)

    assert pytest.raises(AssertionError, lambda: t5 - Time.from_sec(2.0, accuracy=2000))
