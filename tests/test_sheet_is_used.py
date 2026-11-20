"""Every sheet field is read, and its value is not copied as a literal."""

import dataclasses
import pathlib
import re
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import opcfilm.config as c

ROOTS = [pathlib.Path("src"), pathlib.Path("analysis"), pathlib.Path("MD")]
CODE = "\n".join(
    path.read_text(encoding="utf-8", errors="replace")
    for root in ROOTS
    for path in root.rglob("*.py")
    if "config.py" not in path.name
)
ALLOW_LITERALS = {0.5, 1.0, 2.0, 0.0}


def fields():
    """Yield every sheet field.

    Yields
    ------
    tuple
        Section name, field name, and the value stored on ``CONFIG``.
    """
    for section in dataclasses.fields(c.CONFIG):
        body = getattr(c.CONFIG, section.name)
        if dataclasses.is_dataclass(body):
            for field in dataclasses.fields(body):
                yield section.name, field.name, getattr(body, field.name)


class SheetIsUsed(unittest.TestCase):
    def test_every_field_is_read(self):
        missing = [
            f"{section}.{name}"
            for section, name, _value in fields()
            if not re.search(rf"\.{section}\.{name}\b", CODE)
        ]
        self.assertFalse(missing, f"Sheet fields never read: {missing}")

    def test_no_literal_copies(self):
        hits = []
        for section, name, value in fields():
            values = value if isinstance(value, (list, tuple)) else [value]
            for item in values:
                if isinstance(item, float) and item not in ALLOW_LITERALS:
                    if re.search(rf"(default\s*=\s*|=\s*){re.escape(repr(item))}\b", CODE):
                        hits.append(f"{section}.{name}={item}")
        self.assertFalse(hits, f"Literal copies of sheet values: {hits}")


if __name__ == "__main__":
    unittest.main()
