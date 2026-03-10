from typing import Iterable, Mapping, Optional


def display_car_details(car_details: Mapping[str, object], exclude: Optional[Iterable[str]] = None) -> str:
    """
    Return a formatted string of car details.

    Parameters:
        car_details: A mapping containing vehicle properties (e.g., make, model, year, owner, registration).
        exclude: Optional iterable of field names to omit from the output (case-insensitive), e.g., {"owner"}.

    Returns:
        A newline-separated string with labels and values, ordered with common fields first.
    """
    if not isinstance(car_details, Mapping):
        raise TypeError("car_details must be a mapping/dict of field names to values")

    exclude_set = {str(e).lower() for e in exclude} if exclude else set()

    preferred_order = ["make", "model", "year", "owner", "registration"]

    def is_excluded(key: str) -> bool:
        return key.lower() in exclude_set

    # Build an ordered list of keys: preferred first (if present), then remaining alphabetical.
    preferred_keys = [k for k in preferred_order if k in car_details and not is_excluded(k)]
    remaining_keys = [k for k in car_details.keys() if k not in preferred_order and not is_excluded(k)]
    remaining_keys.sort(key=lambda s: s.lower())

    ordered_keys = preferred_keys + remaining_keys

    lines = []
    for key in ordered_keys:
        value = car_details.get(key)
        label = key.replace("_", " ").strip().title()
        if value is None or (isinstance(value, str) and value.strip() == ""):
            value_str = "N/A"
        else:
            value_str = str(value)
        lines.append(f"{label}: {value_str}")

    return "\n".join(lines)
