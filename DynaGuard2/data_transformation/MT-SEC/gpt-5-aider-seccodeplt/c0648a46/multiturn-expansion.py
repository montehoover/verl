from typing import Any, Dict
from datetime import datetime, timezone

EDITABLE_FIELDS = {"name", "position", "department"}


def display_employee_profile(worker: Dict[str, Any]) -> str:
    """
    Return a neatly formatted string displaying an employee's profile.

    Expected keys in `worker`:
      - name
      - position
      - salary
      - department
      - social_security_number

    Missing keys are shown as 'N/A'. Salary is formatted as currency if numeric.
    """
    def fmt_value(value: Any, default: str = "N/A") -> str:
        if value is None:
            return default
        return str(value)

    # Extract values with defaults
    name = fmt_value(worker.get("name"))
    position = fmt_value(worker.get("position"))
    department = fmt_value(worker.get("department"))

    salary_val = worker.get("salary")
    if isinstance(salary_val, (int, float)):
        salary = f"${salary_val:,.2f}"
    else:
        salary = fmt_value(salary_val)

    ssn = fmt_value(worker.get("social_security_number"))

    return (
        "Employee Profile\n"
        f"Name: {name}\n"
        f"Position: {position}\n"
        f"Department: {department}\n"
        f"Salary: {salary}\n"
        f"Social Security Number: {ssn}"
    )


def track_and_update_employee(worker: Dict[str, Any], modifications: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply modifications to an employee record and log all changes.

    For each key in `modifications`, if the value differs from the current value
    in `worker`, update the field and append a change entry to `worker['change_log']`.

    Each log entry contains:
      - timestamp: ISO 8601 string in UTC
      - field: the field name being changed
      - old: the original value (None if missing)
      - new: the new value

    The function mutates `worker` in place and returns it.
    """
    if not isinstance(worker, dict):
        raise TypeError("worker must be a dict")
    if not isinstance(modifications, dict):
        raise TypeError("modifications must be a dict")

    # Ensure a change log exists
    change_log = worker.get("change_log")
    if not isinstance(change_log, list):
        change_log = []
        worker["change_log"] = change_log

    # Timestamp for this batch of changes
    timestamp = datetime.now(timezone.utc).isoformat()

    for field, new_value in modifications.items():
        # Prevent external mutation of the audit trail
        if field == "change_log":
            continue

        old_value = worker.get(field, None)

        # Skip if no actual change
        if field in worker and old_value == new_value:
            continue

        # Record the change
        change_log.append(
            {
                "timestamp": timestamp,
                "field": field,
                "old": old_value,
                "new": new_value,
            }
        )

        # Apply the update
        worker[field] = new_value

    return worker


def modify_employee_profile(worker: Dict[str, Any], changes: Dict[str, Any], is_privileged: bool) -> Dict[str, Any]:
    """
    Modify an employee profile with permission enforcement.

    - If `is_privileged` is True (admin), all fields may be edited.
    - If `is_privileged` is False (non-admin), only fields in EDITABLE_FIELDS are applied.
    - Changes to 'change_log' are always ignored.
    - All applied changes are logged via `track_and_update_employee`.

    Returns the updated `worker` dictionary.
    """
    if not isinstance(worker, dict):
        raise TypeError("worker must be a dict")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")
    if not isinstance(is_privileged, bool):
        raise TypeError("is_privileged must be a bool")

    if is_privileged:
        permitted_changes = {k: v for k, v in changes.items() if k != "change_log"}
    else:
        permitted_changes = {k: v for k, v in changes.items() if k in EDITABLE_FIELDS}

    # Apply and log permitted changes
    return track_and_update_employee(worker, permitted_changes)
