import logging
from datetime import datetime
from typing import Iterable, Optional, Any, Dict

# Module-level logger setup to ensure changes are captured.
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

EDITABLE_FIELDS = {"name", "position", "department"}


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


def update_employee_record(
    worker: Dict[str, Any], modifications: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update an employee dictionary in-place with provided modifications,
    logging each change (old and new values) and recording it in an internal change log.

    The function:
      - Applies only actual changes (when the new value differs from the old value or is newly added).
      - Logs each change via the module logger at INFO level.
      - Appends each change to worker['_change_log'] with a timestamp.

    Args:
        worker: The employee dictionary to update.
        modifications: A dictionary of field -> new_value pairs to apply.

    Returns:
        The updated employee dictionary (same object as provided).
    """
    if not isinstance(worker, dict):
        raise TypeError("worker must be a dict")
    if not isinstance(modifications, dict):
        raise TypeError("modifications must be a dict")

    CHANGE_LOG_KEY = "_change_log"
    MISSING = object()

    # Ensure a change log list exists on the worker record.
    change_log = worker.setdefault(CHANGE_LOG_KEY, [])

    for field, new_value in modifications.items():
        # Prevent external mutation of the change log structure.
        if field == CHANGE_LOG_KEY:
            logger.warning(
                "Attempted to modify reserved field '%s' directly; ignoring.", CHANGE_LOG_KEY
            )
            continue

        old_value = worker.get(field, MISSING)

        # Only record a change if the value is different or the field is new.
        has_changed = (old_value is MISSING) or (old_value != new_value)
        if not has_changed:
            continue

        # Apply the update.
        worker[field] = new_value

        # Build the change entry.
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "field": field,
            "old": None if old_value is MISSING else old_value,
            "new": new_value,
        }

        # Append to the in-record change log and emit via logger.
        change_log.append(entry)
        logger.info(
            "Employee change - %s: %r -> %r",
            field,
            entry["old"],
            entry["new"],
        )

    return worker


def modify_employee_data(
    worker: Dict[str, Any],
    alterations: Dict[str, Any],
    admin_privileges: bool,
) -> Dict[str, Any]:
    """
    Modify employee data with role-based permissions.

    Admin users can modify any field.
    Regular users can only modify fields in EDITABLE_FIELDS.
    Applies updates via update_employee_record to log each change.

    Args:
        worker: The employee dictionary to modify.
        alterations: Proposed field -> value modifications.
        admin_privileges: True if the acting user has admin privileges.

    Returns:
        The updated employee dictionary.
    """
    if not isinstance(worker, dict):
        raise TypeError("worker must be a dict")
    if not isinstance(alterations, dict):
        raise TypeError("alterations must be a dict")
    if not isinstance(admin_privileges, bool):
        raise TypeError("admin_privileges must be a bool")

    if admin_privileges:
        allowed_changes = alterations
    else:
        allowed_changes = {k: v for k, v in alterations.items() if k in EDITABLE_FIELDS}
        denied_fields = sorted(set(alterations.keys()) - set(allowed_changes.keys()))
        if denied_fields:
            logger.warning(
                "Permission denied: attempted to modify restricted fields: %s",
                ", ".join(denied_fields),
            )

    if not allowed_changes:
        return worker

    return update_employee_record(worker, allowed_changes)
