"""Enzyme Commission (EC) number parsing and hierarchical expansion.

EC numbers are 4-level hierarchical: "1.2.3.4" means class 1, subclass 2,
sub-subclass 3, specific enzyme 4. The bio-sae hierarchical tier feeds
the SAE every prefix so it can recover both broad class membership
("oxidoreductase") and the leaf annotation.

Wildcards "1.-.-.-." are valid EC strings (used when only the broad
class is known); these are tolerated and contribute only the levels they
specify.
"""

from __future__ import annotations

import re
from typing import Iterable


_EC_PATTERN = re.compile(r"^\d+(\.\d+){0,3}(\.\-)*$|^\d+(\.(?:\d+|\-)){0,3}$")


def is_valid_ec(ec: str) -> bool:
    if not _EC_PATTERN.match(ec):
        return False
    parts = ec.split(".")
    if not parts or not parts[0].isdigit():
        return False
    return len(parts) <= 4


def expand(ec: str) -> set[str]:
    """Return the set of prefixes for an EC number.

        "1.2.3.4"   -> {"1", "1.2", "1.2.3", "1.2.3.4"}
        "1.2.-.-"   -> {"1", "1.2"}
        "1.-.-.-"   -> {"1"}
    """
    if not is_valid_ec(ec):
        return set()
    parts = ec.split(".")
    out: set[str] = set()
    prefix: list[str] = []
    for p in parts:
        if p in ("-", "*", ""):
            break
        prefix.append(p)
        out.add(".".join(prefix))
    return out


def expand_many(ec_strings: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for ec in ec_strings:
        out.update(expand(ec))
    return out


def class_of(ec: str) -> str | None:
    """Return the top-level class digit as a string, or None if invalid."""
    if not is_valid_ec(ec):
        return None
    return ec.split(".", 1)[0]


EC_TOP_CLASSES = {
    "1": "oxidoreductase",
    "2": "transferase",
    "3": "hydrolase",
    "4": "lyase",
    "5": "isomerase",
    "6": "ligase",
    "7": "translocase",
}
