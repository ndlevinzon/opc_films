"""Numeric sheet fields are numbers, even when YAML 1.1 reads them as text."""

import dataclasses

from opcfilm import config as c


def test_numeric_fields_are_numbers():
    for section in dataclasses.fields(c.CONFIG):
        body = getattr(c.CONFIG, section.name)
        if not dataclasses.is_dataclass(body):
            continue
        for field in dataclasses.fields(body):
            value = getattr(body, field.name)
            if str(field.type) in ("float", "<class 'float'>"):
                assert isinstance(value, float), f"{section.name}.{field.name} = {value!r}"


def test_unsigned_exponent_is_converted():
    assert c._coerce("6.02214076e23", "float") == 6.02214076e23
    assert c._coerce(("1e3", 2.0), "tuple[float, ...]") == (1000.0, 2.0)
    assert c._coerce("abc", "str") == "abc"
