from typing import Any, Iterable, Mapping, Optional


def view_vehicle_details(
    car: Mapping[str, Any],
    fields_to_exclude: Optional[Iterable[str]] = None,
) -> str:
    """
    Return a formatted string presenting vehicle details.

    Parameters:
        car: A mapping representing a vehicle, expected keys may include:
             make, model, year, owner, registration, etc.
        fields_to_exclude: An optional iterable of field names to hide from the output
                           (e.g., {"registration"}). Matching is case-insensitive.

    Returns:
        A neatly formatted, multi-line string of vehicle details.
    """
    if not isinstance(car, Mapping):
        raise TypeError("car must be a mapping/dict of vehicle attributes")

    exclude = {f.lower() for f in fields_to_exclude} if fields_to_exclude else set()

    # Preferred display order for common fields; any remaining fields will follow.
    preferred_order = ["make", "model", "year", "owner", "registration"]

    def is_included(key: str) -> bool:
        return key.lower() not in exclude

    ordered_keys = [k for k in preferred_order if k in car and is_included(k)]
    remaining_keys = [k for k in car.keys() if k not in preferred_order and is_included(k)]
    remaining_keys.sort(key=lambda s: s.lower())

    lines = ["Vehicle Details:"]
    count = 0

    for key in ordered_keys + remaining_keys:
        label = key.replace("_", " ").strip().capitalize()
        value = car.get(key)
        lines.append(f"- {label}: {value}")
        count += 1

    if count == 0:
        lines.append("- No details to display")

    return "\n".join(lines)


__all__ = ["view_vehicle_details"]
