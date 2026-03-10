import logging
from typing import Iterable, Optional, Any, Dict, Set


__all__ = ["get_employee_details", "update_employee_with_logging", "adjust_employee_details"]

logger = logging.getLogger(__name__)
_MISSING = object()

# Non-admin users may only edit the following fields
EDITABLE_FIELDS: Set[str] = {"name", "position", "department"}


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


def update_employee_with_logging(person: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update an employee dictionary with given changes and log each modification.

    Args:
        person: The employee dictionary to update.
        changes: A dictionary of keys and new values to set on the employee.

    Returns:
        The updated employee dictionary (the same object, modified in place).

    Logging:
        Emits an INFO log record for each changed field with old and new values.
        New fields are logged as change_type='add', existing fields as 'update'.
    """
    if not isinstance(person, dict):
        raise TypeError("person must be a dict")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")

    # Capture an identifier for traceability, if available, before applying changes
    identifier = (
        person.get("id")
        or person.get("employee_id")
        or person.get("email")
        or person.get("name")
        or "<unknown-employee>"
    )

    for key, new_value in changes.items():
        old_value = person.get(key, _MISSING)
        is_new_key = old_value is _MISSING
        changed = is_new_key or old_value != new_value

        if not changed:
            # No modification; skip logging to avoid noise.
            continue

        person[key] = new_value  # Apply the update

        # Convert sentinel to None for clearer logs
        old_for_log = None if is_new_key else old_value
        change_type = "add" if is_new_key else "update"

        # Structured, readable log message
        logger.info(
            "Employee update (%s): field=%s old=%r new=%r change_type=%s",
            identifier,
            key,
            old_for_log,
            new_value,
            change_type,
        )

    return person


def adjust_employee_details(person: Dict[str, Any], alterations: Dict[str, Any], has_admin_rights: bool) -> Dict[str, Any]:
    """
    Adjust employee details with role-based permissions.

    Admin users can modify any field. Non-admin users can update only fields in EDITABLE_FIELDS.
    The function updates the given person dictionary (in place) and returns it.

    Args:
        person: The employee profile dictionary.
        alterations: The desired changes to apply.
        has_admin_rights: Whether the caller has admin privileges.

    Returns:
        The updated employee profile dictionary.
    """
    if not isinstance(person, dict):
        raise TypeError("person must be a dict")
    if not isinstance(alterations, dict):
        raise TypeError("alterations must be a dict")
    if not isinstance(has_admin_rights, bool):
        raise TypeError("has_admin_rights must be a bool")

    if has_admin_rights:
        allowed_changes = alterations
        blocked_keys = []
    else:
        allowed_changes: Dict[str, Any] = {}
        blocked_keys = []
        for k, v in alterations.items():
            if k in EDITABLE_FIELDS:
                allowed_changes[k] = v
            else:
                blocked_keys.append(k)

    # Log any blocked keys for traceability
    if blocked_keys:
        identifier = (
            person.get("id")
            or person.get("employee_id")
            or person.get("email")
            or person.get("name")
            or "<unknown-employee>"
        )
        logger.warning(
            "Permission denied for fields on employee (%s): %s",
            identifier,
            ", ".join(sorted(set(map(str, blocked_keys)))),
        )

    # Apply allowed changes with detailed logging
    return update_employee_with_logging(person, allowed_changes)
