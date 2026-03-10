from typing import Any, Dict, Iterable, Optional
import json


def _humanize_label(key: str) -> str:
    """Convert a dict key to a human-friendly label."""
    return key.replace("_", " ").strip().title()


def _stringify_value(value: Any) -> str:
    """Convert values to a readable string representation."""
    if isinstance(value, (dict, list, tuple, set)):
        try:
            return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
        except Exception:
            return str(value)
    return str(value)


def display_vehicle_info(vehicle: Dict[str, Any], fields_to_exclude: Optional[Iterable[str]] = None) -> str:
    """
    Return a neatly formatted string with vehicle details.

    Parameters:
        vehicle: Dictionary representing a vehicle (e.g., make, model, year, owner, registration).
        fields_to_exclude: Optional iterable of field names to omit (e.g., {"owner"}). Case-insensitive.

    Returns:
        A formatted multi-line string with the vehicle's details. If no displayable fields are present,
        a minimal message is returned.
    """
    if not isinstance(vehicle, dict):
        raise TypeError("vehicle must be a dict")

    # Normalize excluded fields (case-insensitive).
    if fields_to_exclude is None:
        excluded = set()
    elif isinstance(fields_to_exclude, str):
        excluded = {fields_to_exclude.lower()}
    else:
        excluded = {str(k).lower() for k in fields_to_exclude}

    # Preferred display order for common fields.
    preferred_order = ["make", "model", "year", "registration", "owner"]
    label_overrides = {
        "make": "Make",
        "model": "Model",
        "year": "Year",
        "registration": "Registration",
        "owner": "Owner",
    }

    # Build the ordered list of (key, value) pairs to display.
    items = []

    # Add preferred keys first.
    seen = set()
    for key in preferred_order:
        if key in vehicle:
            if key.lower() in excluded:
                continue
            value = vehicle.get(key)
            if value is None or value == "":
                continue
            items.append((key, value))
            seen.add(key)

    # Add remaining keys in alphabetical order (case-insensitive).
    remaining_keys = [
        k for k in vehicle.keys()
        if k not in seen and k.lower() not in excluded and vehicle.get(k) not in (None, "")
    ]
    remaining_keys.sort(key=lambda x: x.lower())
    for key in remaining_keys:
        items.append((key, vehicle[key]))

    # If nothing to show, return a minimal message.
    if not items:
        return "Vehicle Details: No data available."

    # Format output.
    lines = ["Vehicle Details"]
    for key, value in items:
        label = label_overrides.get(key.lower(), _humanize_label(key))
        lines.append(f"- {label}: {_stringify_value(value)}")

    return "\n".join(lines)


__all__ = ["display_vehicle_info"]
