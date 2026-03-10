from typing import Any, Dict, Iterable, Optional


DEFAULT_FIELD_ORDER = ("make", "model", "year", "registration", "owner")


def _humanize_key(key: str) -> str:
    special = {
        "vin": "VIN",
        "id": "ID",
        "license_plate": "License Plate",
        "number_plate": "Number Plate",
    }
    lowered = key.lower()
    if lowered in special:
        return special[lowered]
    parts = lowered.replace("-", "_").split("_")
    return " ".join(p.capitalize() for p in parts if p)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, bytes)) and value == "":
        return True
    if isinstance(value, (list, tuple, set, dict)) and len(value) == 0:
        return True
    return False


def display_vehicle_info(auto: Dict[str, Any], fields_to_hide: Optional[Iterable[str]] = None) -> str:
    """
    Return a formatted multi-line string of vehicle details.

    Parameters:
    - auto: dict with vehicle attributes (e.g., make, model, year, owner, registration, etc.)
    - fields_to_hide: iterable of keys (case-insensitive) to exclude from output, e.g., {"owner"}.

    Returns:
    - A string with one "Label: Value" per line, skipping hidden or empty fields.
    """
    if not isinstance(auto, dict):
        raise TypeError("auto must be a dict")

    hide = {k.lower() for k in (fields_to_hide or [])}
    lines: list[str] = []
    seen_keys: set[str] = set()

    # Emit known fields in a preferred order
    for key in DEFAULT_FIELD_ORDER:
        real_key = next((rk for rk in auto.keys() if rk.lower() == key), None)
        if not real_key:
            continue
        if real_key.lower() in hide:
            seen_keys.add(real_key)
            continue
        val = auto.get(real_key)
        if _is_empty(val):
            seen_keys.add(real_key)
            continue
        lines.append(f"{_humanize_key(real_key)}: {val}")
        seen_keys.add(real_key)

    # Emit any remaining fields not in the preferred order
    for rk, val in auto.items():
        if rk in seen_keys:
            continue
        if rk.lower() in hide or _is_empty(val):
            continue
        lines.append(f"{_humanize_key(rk)}: {val}")

    return "\n".join(lines)
