from typing import Any, Dict
import logging

__all__ = ["get_employee_record", "update_and_log_employee_info"]


def get_employee_record(employee_record: Dict[str, Any], hide_sensitive: bool = True) -> str:
    """
    Format an employee record for display.

    Parameters:
        employee_record: dict containing employee details. Common keys include:
            - name
            - position
            - salary
            - department
            - social_security_number
        hide_sensitive: if True, omits sensitive fields like social_security_number from the output.

    Returns:
        A formatted string suitable for display.
    """
    if not isinstance(employee_record, dict):
        raise TypeError("employee_record must be a dictionary")

    labels = {
        "name": "Name",
        "position": "Position",
        "salary": "Salary",
        "department": "Department",
        "social_security_number": "Social Security Number",
    }

    # Define the display order explicitly
    field_order = [
        "name",
        "position",
        "salary",
        "department",
        "social_security_number",
    ]

    sensitive_fields = {"social_security_number"}

    def _format_salary(value: Any) -> str:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return str(value)
        # Format with thousands separator; drop decimals if whole number
        if num.is_integer():
            return f"{int(num):,}"
        return f"{num:,.2f}"

    lines = []
    for key in field_order:
        if hide_sensitive and key in sensitive_fields:
            continue

        value = employee_record.get(key, "N/A")
        if key == "salary" and value != "N/A":
            value = _format_salary(value)

        label = labels.get(key, key.capitalize())
        lines.append(f"{label}: {value}")

    return "\n".join(lines)


def update_and_log_employee_info(
    employee_record: Dict[str, Any],
    updates: Dict[str, Any],
    logger: Any,
) -> Dict[str, Any]:
    """
    Update an employee record and log each change with before/after values.

    Parameters:
        employee_record: The record to update (will be mutated).
        updates: A dict of fields and their new values.
        logger: Either a callable that accepts a single message string,
                or a logging.Logger-like object with an .info(str) or .log(level, str) method.

    Returns:
        The updated employee_record.
    """
    if not isinstance(employee_record, dict):
        raise TypeError("employee_record must be a dictionary")
    if not isinstance(updates, dict):
        raise TypeError("updates must be a dictionary")

    # Resolve a logging function
    log_callable = _resolve_logger(logger)

    sensitive_fields = {"social_security_number"}

    def _mask_if_sensitive(key: str, value: Any) -> str:
        if key in sensitive_fields:
            return _mask_value(value)
        return repr(value)

    MISSING = object()

    for key, new_value in updates.items():
        old_present = key in employee_record
        old_value = employee_record.get(key, MISSING)

        # Determine if a change occurred
        changed = (old_value is MISSING) or (old_value != new_value)

        if changed:
            before_str = "<missing>" if old_value is MISSING else _mask_if_sensitive(key, old_value)
            after_str = _mask_if_sensitive(key, new_value)

            if old_value is MISSING:
                log_callable(f"Added field '{key}': from {before_str} to {after_str}")
            else:
                log_callable(f"Updated field '{key}': from {before_str} to {after_str}")

            # Apply the update
            employee_record[key] = new_value

    return employee_record


def _resolve_logger(logger: Any):
    """
    Return a callable that logs a message string.
    Supports:
      - a simple callable (message: str) -> None
      - logging.Logger instances (uses .info)
      - objects with .info(str) or .log(level, str)
    """
    if callable(logger):
        return logger
    # logging.Logger or similar with .info
    info = getattr(logger, "info", None)
    if callable(info):
        return info
    # Fallback to .log with INFO level
    log = getattr(logger, "log", None)
    if callable(log):
        return lambda msg: log(logging.INFO, msg)

    raise TypeError(
        "logger must be a callable or an object with .info(str) or .log(level, str)"
    )


def _mask_value(value: Any) -> str:
    """
    Mask a sensitive value, preserving only the last 4 characters.
    Non-string values are converted to string first.
    """
    s = str(value)
    if len(s) <= 4:
        return "*" * len(s)
    masked_len = len(s) - 4
    return "*" * masked_len + s[-4:]
