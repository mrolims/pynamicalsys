"""
No compiled function uses Numba's on-disk cache. A cached function that takes
another compiled function as an argument adds new cache entries in every
process and can later load a stale one, failing with "underlying object has
vanished".
"""

import pathlib
import re

SOURCE = pathlib.Path(__file__).resolve().parents[2] / "src" / "pynamicalsys"


def test_no_function_is_compiled_with_cache():
    offenders = [
        f"{path.relative_to(SOURCE)}:{number}"
        for path in sorted(SOURCE.rglob("*.py"))
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if re.search(r"\bcache\s*=\s*True", line)
    ]
    assert offenders == []
