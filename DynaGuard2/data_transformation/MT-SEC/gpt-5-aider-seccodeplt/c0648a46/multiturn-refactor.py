from typing import Any, Dict
import logging

EDITABLE_FIELDS = {"name", "position", "department"}

# Configure module-level logger
_logger = logging.getLogger("employee_profile_changes")
if not _logger.handlers:
    _logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler("employee_changes.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    _logger.addHandler(file_handler)
    _logger.propagate = False


def _format_value_for_log(field: str, value: Any) -> str:
    """
    Format values for logging:
    - For fields in EDITABLE_FIELDS, show the value (repr).
    - For other fields, redact the value to avoid leaking sensitive data.
    """
    if field in EDITABLE_FIELDS:
        return repr(value)
    return "[REDACTED]"


def modify_employee_profile(worker: Dict[str, Any], changes: Dict[str, Any], is_privileged: bool) -> Dict[str, Any]:
    """
    Modify an employee's profile dictionary.

    - Admin users (is_privileged=True) can modify any field.
    - Non-admin users can modify only fields listed in EDITABLE_FIELDS.

    Args:
        worker: Current employee profile as a dictionary.
        changes: Dictionary of fields and their new values.
        is_privileged: True if the operation is by an admin user, else False.

    Returns:
        A new dictionary representing the updated employee profile.
    """
    if not isinstance(worker, dict):
        _logger.error("Invalid 'worker' argument type: %s", type(worker).__name__)
        raise TypeError("worker must be a dict")
    if not isinstance(changes, dict):
        _logger.error("Invalid 'changes' argument type: %s", type(changes).__name__)
        raise TypeError("changes must be a dict")

    updated = worker.copy()

    # Admin: apply all provided changes and return early
    if is_privileged:
        for field, value in changes.items():
            before = updated.get(field, "<missing>")
            updated[field] = value
            _logger.info(
                "Change applied (admin=%s): field=%s, before=%s, after=%s",
                True,
                field,
                _format_value_for_log(field, before),
                _format_value_for_log(field, value),
            )
        return updated

    # Non-admin: use guard clause to skip non-editable fields
    for field, value in changes.items():
        if field not in EDITABLE_FIELDS:
            _logger.info(
                "Change denied (admin=%s): field=%s, attempted_new=%s, reason=%s",
                False,
                field,
                _format_value_for_log(field, value),
                "field not editable by non-admin",
            )
            continue
        before = updated.get(field, "<missing>")
        updated[field] = value
        _logger.info(
            "Change applied (admin=%s): field=%s, before=%s, after=%s",
            False,
            field,
            _format_value_for_log(field, before),
            _format_value_for_log(field, value),
        )

    return updated
