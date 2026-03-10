from typing import Iterable, Optional, Any, Dict


def display_employee_details(
    worker: Dict[str, Any],
    fields_to_hide: Optional[Iterable[str]] = None,
) -> str:
    """
    Return a formatted string of employee details.

    Args:
        worker: A dictionary representing an employee. Expected keys may include:
                - name
                - position
                - salary
                - department
                - social_security_number
                Additional keys are supported and will be displayed unless hidden.
        fields_to_hide: Iterable of field names to exclude from the output.
                        Defaults to hiding 'social_security_number'.

    Returns:
        A multi-line string with "Label: value" per line, excluding hidden fields.
    """
    if fields_to_hide is None:
        fields_to_hide = {"social_security_number"}
    else:
        fields_to_hide = set(fields_to_hide)

    # Preferred order for known fields; any extra fields are appended afterward.
    preferred_order = [
        "name",
        "position",
        "department",
        "salary",
        "social_security_number",
    ]

    lines = []

    def format_label(key: str) -> str:
        return key.replace("_", " ").strip().title()

    def format_value(value: Any) -> str:
        if isinstance(value, (int, float)) and "salary" in current_key:
            # Try to format salary with thousands separator; keep decimals if present.
            if isinstance(value, int):
                return f"${value:,}"
            else:
                return f"${value:,.2f}"
        return str(value)

    # First, add fields in preferred order if present and not hidden.
    for key in preferred_order:
        if key in worker and key not in fields_to_hide:
            current_key = key  # used inside format_value
            value = worker[key]
            lines.append(f"{format_label(key)}: {format_value(value)}")

    # Then, include any remaining fields that aren't hidden and not already added.
    added = set(k for k in preferred_order if k in worker and k not in fields_to_hide)
    for key, value in worker.items():
        if key in added or key in fields_to_hide:
            continue
        current_key = key  # used inside format_value
        lines.append(f"{format_label(key)}: {format_value(value)}")

    return "\n".join(lines)
