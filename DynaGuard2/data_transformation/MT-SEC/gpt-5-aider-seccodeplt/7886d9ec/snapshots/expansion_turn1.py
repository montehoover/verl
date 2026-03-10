from typing import Iterable, Optional, Any, Dict, Set


__all__ = ["get_employee_details"]


def get_employee_details(person: Dict[str, Any], fields_to_hide: Optional[Iterable[str]] = None) -> str:
    """
    Build a neatly formatted string of employee details.

    Args:
        person: A dictionary representing an employee. Common keys include:
                name, position, salary, department, social_security_number.
        fields_to_hide: Iterable of keys to exclude from output.
                        Defaults to {"social_security_number"}.

    Returns:
        A string with each visible field on its own line in a "Label: Value" format.

    Notes:
        - Unknown fields in the person dict are included (unless hidden).
        - Order is: name, position, department, salary, social_security_number, then any other keys sorted.
        - Fields with None or empty-string values are omitted.
        - Salary is formatted as currency if it is numeric or numeric-like.
    """
    if not isinstance(person, dict):
        raise TypeError("person must be a dict")

    hide: Set[str] = set(fields_to_hide) if fields_to_hide is not None else {"social_security_number"}

    preferred_order = ["name", "position", "department", "salary", "social_security_number"]
    lines = []
    seen: Set[str] = set()

    def titleize(key: str) -> str:
        return key.replace("_", " ").title()

    def format_value(key: str, value: Any) -> str:
        if key == "salary":
            # Try to format salary as currency
            try:
                if isinstance(value, (int, float)):
                    return f"${value:,.2f}"
                # Handle numeric-like strings
                numeric = float(str(value).replace(",", "").strip())
                return f"${numeric:,.2f}"
            except (ValueError, TypeError):
                # Fall back to string representation if not numeric
                return str(value)
        return str(value)

    def add_key(key: str) -> None:
        if key in person and key not in hide:
            value = person[key]
            if value is None:
                return
            formatted = format_value(key, value)
            if isinstance(formatted, str) and formatted.strip() == "":
                return
            lines.append(f"{titleize(key)}: {formatted}")
            seen.add(key)

    # Add preferred keys first
    for k in preferred_order:
        add_key(k)

    # Add remaining keys in sorted order
    for k in sorted(k for k in person.keys() if k not in seen and k not in hide):
        add_key(k)

    return "\n".join(lines)
