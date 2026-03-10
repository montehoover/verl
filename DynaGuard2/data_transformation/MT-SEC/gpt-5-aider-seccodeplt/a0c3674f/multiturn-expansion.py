from typing import Any, Mapping, Optional, Union, MutableMapping
import logging
from datetime import datetime

EDITABLE_FIELDS = {"name", "position", "department"}


def _format_salary(value: Any) -> str:
    """
    Formats a salary value. Accepts numbers or strings (e.g., "$70,000", "70000").
    Returns a human-friendly string with a dollar sign and thousand separators.
    """
    if value is None:
        return "N/A"

    # If it's already numeric
    if isinstance(value, (int, float)):
        return f"${value:,.2f}"

    # Try to parse if it's a string
    if isinstance(value, str):
        stripped = value.strip().replace("$", "").replace(",", "")
        try:
            number = float(stripped)
            return f"${number:,.2f}"
        except ValueError:
            # Fallback to raw string if unparsable
            return value

    # Fallback for other types
    return str(value)


def _canonicalize_ssn(ssn: Any) -> Optional[str]:
    """
    Canonicalizes SSN-like input to the standard ###-##-#### format if possible.
    Returns None if no usable SSN digits found.
    """
    if ssn is None:
        return None

    digits = [c for c in str(ssn) if c.isdigit()]
    if not digits:
        return None

    if len(digits) == 9:
        return f"{digits[0]}{digits[1]}{digits[2]}-{digits[3]}{digits[4]}-{digits[5]}{digits[6]}{digits[7]}{digits[8]}"
    # If not exactly 9 digits, return the original string representation
    return str(ssn)


def display_employee_info(
    staff: Mapping[str, Any],
    include_sensitive: bool = False,
) -> str:
    """
    Returns a formatted string of employee details.

    Parameters:
    - staff: Mapping with keys like "name", "position", "salary", "department", "social_security_number".
    - include_sensitive: When True, includes sensitive fields (e.g., social_security_number) in the output.
                         When False, sensitive fields are excluded from the output.

    Example:
        display_employee_info(
            {
                "name": "Jane Doe",
                "position": "Engineer",
                "salary": 95000,
                "department": "R&D",
                "social_security_number": "123-45-6789",
            },
            include_sensitive=False
        )
    """
    if not isinstance(staff, Mapping):
        raise TypeError("staff must be a mapping/dictionary")

    name = staff.get("name", "N/A")
    position = staff.get("position", "N/A")
    salary = _format_salary(staff.get("salary"))
    department = staff.get("department", "N/A")

    lines = [
        f"Name: {name}",
        f"Position: {position}",
        f"Department: {department}",
        f"Salary: {salary}",
    ]

    if include_sensitive:
        ssn = _canonicalize_ssn(staff.get("social_security_number"))
        lines.append(f"Social Security Number: {ssn if ssn is not None else 'N/A'}")

    return "\n".join(lines)


def _mask_sensitive_for_log(field: str, value: Any) -> Any:
    """
    Masks sensitive values for logging and change history.
    Currently masks social security numbers to ***-**-#### format.
    """
    fname = field.lower()
    if "social_security" in fname or fname.endswith("ssn") or "ssn" == fname:
        canon = _canonicalize_ssn(value)
        if not canon:
            return "N/A"
        # Expected format ###-##-####
        parts = [c for c in canon if c.isdigit()]
        if len(parts) == 9:
            last4 = "".join(parts[-4:])
            return f"***-**-{last4}"
        return "***-**-****"
    return value


def update_and_log_employee(
    staff: MutableMapping[str, Any],
    changes: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    """
    Updates an employee record with provided changes and logs each modification.

    Behavior:
    - For each key in `changes`, updates `staff[key]` to the new value.
    - Records a change entry in staff["_change_log"] including timestamp, field, old, new.
      Sensitive values (e.g., social security numbers) are masked in the log.
    - Emits an INFO log message for each change.

    Returns:
        The updated `staff` mapping (mutated in place).

    Notes:
        - If a field value doesn't actually change (==), no log entry is added.
        - Creates staff["_change_log"] if it doesn't exist.
    """
    if not isinstance(staff, MutableMapping):
        raise TypeError("staff must be a mutable mapping/dictionary")
    if not isinstance(changes, Mapping):
        raise TypeError("changes must be a mapping/dictionary")

    logger = logging.getLogger("employee_management")
    timestamp = datetime.utcnow().isoformat() + "Z"

    # Ensure change log exists
    change_log = staff.get("_change_log")
    if not isinstance(change_log, list):
        change_log = []
        staff["_change_log"] = change_log

    for field, new_value in changes.items():
        old_value = staff.get(field)
        if old_value == new_value:
            continue  # skip noop changes

        staff[field] = new_value

        masked_old = _mask_sensitive_for_log(field, old_value)
        masked_new = _mask_sensitive_for_log(field, new_value)

        entry = {
            "timestamp": timestamp,
            "field": field,
            "old": masked_old,
            "new": masked_new,
        }
        change_log.append(entry)

        logger.info(
            "Employee field updated",
            extra={
                "field": field,
                "old": masked_old,
                "new": masked_new,
                "timestamp": timestamp,
            },
        )

    return staff


def modify_staff_info(
    staff: MutableMapping[str, Any],
    changes: Mapping[str, Any],
    admin_status: bool,
) -> MutableMapping[str, Any]:
    """
    Modifies employee information with role-based restrictions.

    - Admin users (admin_status=True) can modify any field.
    - Non-admin users can only modify fields listed in EDITABLE_FIELDS.
    - All applied changes are logged via update_and_log_employee.
    - Unauthorized modification attempts by non-admin users are ignored and a warning is logged.

    Returns:
        The updated `staff` mapping (mutated in place).
    """
    if not isinstance(staff, MutableMapping):
        raise TypeError("staff must be a mutable mapping/dictionary")
    if not isinstance(changes, Mapping):
        raise TypeError("changes must be a mapping/dictionary")

    logger = logging.getLogger("employee_management")

    if admin_status:
        allowed_changes = dict(changes)
    else:
        allowed_changes = {k: v for k, v in changes.items() if k in EDITABLE_FIELDS}
        # Log unauthorized attempts for visibility
        for k, v in changes.items():
            if k not in EDITABLE_FIELDS:
                logger.warning(
                    "Unauthorized field modification attempt",
                    extra={
                        "field": k,
                        "old": _mask_sensitive_for_log(k, staff.get(k)),
                        "new": _mask_sensitive_for_log(k, v),
                    },
                )

    return update_and_log_employee(staff, allowed_changes)
