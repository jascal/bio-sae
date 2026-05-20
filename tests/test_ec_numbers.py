from biosae.labels import ec_numbers


def test_valid_ec():
    assert ec_numbers.is_valid_ec("1.2.3.4")
    assert ec_numbers.is_valid_ec("1.2")
    assert ec_numbers.is_valid_ec("1.2.-.-")
    assert ec_numbers.is_valid_ec("7.6.2.1")
    assert not ec_numbers.is_valid_ec("EC1.2.3.4")
    assert not ec_numbers.is_valid_ec("1.2.3.4.5")
    assert not ec_numbers.is_valid_ec("a.b.c.d")


def test_expand_full():
    assert ec_numbers.expand("1.2.3.4") == {"1", "1.2", "1.2.3", "1.2.3.4"}


def test_expand_truncated():
    assert ec_numbers.expand("1.2.-.-") == {"1", "1.2"}
    assert ec_numbers.expand("3.-.-.-") == {"3"}


def test_expand_short_form():
    assert ec_numbers.expand("2.7") == {"2", "2.7"}


def test_expand_many_dedupes():
    out = ec_numbers.expand_many(["1.2.3.4", "1.2.3.5", "1.2.4.1"])
    assert out == {"1", "1.2", "1.2.3", "1.2.3.4", "1.2.3.5", "1.2.4", "1.2.4.1"}


def test_class_of():
    assert ec_numbers.class_of("3.4.21.1") == "3"
    assert ec_numbers.class_of("nonsense") is None
