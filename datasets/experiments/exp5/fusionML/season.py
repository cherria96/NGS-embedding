"""Season-code normalization (design-spec item 9, "The season mapping is
fixed by the user"). Exact-token matching only -- "Sp" must never be
interpreted as "S" via a prefix/substring match."""
from __future__ import annotations

from typing import Optional

SEASON_MAPPING = {"S": 1, "F": 2, "W": 3, "Sp": 4}
SEASON_NAMES = {"S": "summer", "F": "fall", "W": "winter", "Sp": "spring"}
ACCEPTED_NUMERIC = {1, 2, 3, 4}


class SeasonError(ValueError):
    pass


def normalize_season(raw) -> int:
    """Accepts an exact label token (S/F/W/Sp) or an already-numeric 1-4
    code (int, float, or numeral string) and returns the canonical int
    code. Anything else -- including partial/substring matches -- raises."""
    if isinstance(raw, str):
        token = raw.strip()
        if token in SEASON_MAPPING:
            return SEASON_MAPPING[token]
        try:
            numeric = int(token)
        except ValueError:
            raise SeasonError(
                f"season token {raw!r} does not exactly match a configured label "
                f"{list(SEASON_MAPPING)} or a numeral 1-4 (exact match only; "
                f"e.g. 'Sp' must not be interpreted as 'S')."
            )
        if numeric in ACCEPTED_NUMERIC:
            return numeric
        raise SeasonError(f"numeric season {raw!r} is not in {sorted(ACCEPTED_NUMERIC)}")
    if isinstance(raw, (int, float)):
        numeric = int(raw)
        if numeric == raw and numeric in ACCEPTED_NUMERIC:
            return numeric
        raise SeasonError(f"numeric season {raw!r} is not an integer in {sorted(ACCEPTED_NUMERIC)}")
    raise SeasonError(f"unrecognized season value {raw!r} of type {type(raw)}")
