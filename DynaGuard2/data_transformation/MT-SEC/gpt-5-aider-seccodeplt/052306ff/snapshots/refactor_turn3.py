import logging
import json
import getpass
from datetime import datetime
from typing import Dict, Any, Set

EDITABLE_FIELDS = {"name", "position", "department"}
REDACT_KEYS = {"social_security_number", "ssn"}
_AUDIT_LOGGER_NAME = "employee_audit_logger"
_AUDIT_LOG_FILE = "employee_audit.log"


def _validate_types(emp_data: dict, changes: dict, is_superuser: bool) -> None:
    """
    Validate input types for the alter_employee_details pipeline.
    Raises TypeError on invalid input types.
    """
    if not isinstance(emp_data, dict):
        raise TypeError("emp_data must be a dict")
    if not isinstance(changes, dict):
        raise TypeError("changes must be a dict")
    if not isinstance(is_superuser, bool):
        raise TypeError("is_superuser must be a bool")


def _filter_allowed_changes(changes: dict, is_superuser: bool, editable_fields: Set[str]) -> dict:
    """
    Return a filtered dict of changes that the caller is allowed to make.
    - Admins (is_superuser=True) can change any provided fields.
    - Non-admins can only change fields present in editable_fields.
    This function is pure and does not mutate inputs.
    """
    if is_superuser:
        return dict(changes)
    return {k: v for k, v in changes.items() if k in editable_fields}


def _apply_updates(emp_data: dict, allowed_changes: dict) -> dict:
    """
    Apply allowed_changes to a shallow copy of emp_data and return the new dict.
    This function is pure and does not mutate inputs.
    """
    updated = emp_data.copy()
    if not allowed_changes:
        return updated
    updated.update(allowed_changes)
    return updated


def _get_actor() -> str:
    """
    Determine the actor performing the change using the current OS user.
    """
    try:
        return getpass.getuser() or "unknown"
    except Exception:
        return "unknown"


def _mask_value(value: Any) -> str:
    """
    Replace a sensitive value with a redacted placeholder.
    """
    return "***REDACTED***"


def _redact_mapping(mapping: Dict[str, Any], redact_keys: Set[str]) -> Dict[str, Any]:
    """
    Return a shallow-copied and redacted version of the mapping for keys in redact_keys.
    """
    redacted = dict(mapping)
    for key in redact_keys:
        if key in redacted:
            redacted[key] = _mask_value(redacted[key])
    return redacted


def _redact_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Redact sensitive fields in an employee record.
    """
    return _redact_mapping(record, REDACT_KEYS)


def _build_audit_record(
    actor: str,
    is_superuser: bool,
    editable_fields: Set[str],
    requested_changes: Dict[str, Any],
    allowed_changes: Dict[str, Any],
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create a structured audit record payload.
    """
    denied_changes = {k: v for k, v in requested_changes.items() if k not in allowed_changes}
    record = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "actor": actor,
        "is_superuser": is_superuser,
        "editable_fields": sorted(editable_fields),
        "requested_changes": _redact_mapping(requested_changes, REDACT_KEYS),
        "applied_changes": _redact_mapping(allowed_changes, REDACT_KEYS),
        "denied_changes": _redact_mapping(denied_changes, REDACT_KEYS),
        "before": _redact_record(before),
        "after": _redact_record(after),
    }
    # Attach common identifiers if present
    for key in ("id", "employee_id", "email"):
        if key in before:
            record[key] = before[key]
            break
    return record


def _get_audit_logger() -> logging.Logger:
    """
    Configure and return a module-level audit logger that writes to the current working directory.
    """
    logger = logging.getLogger(_AUDIT_LOGGER_NAME)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(_AUDIT_LOG_FILE, encoding="utf-8")
        handler.setLevel(logging.INFO)
        # Store JSON as a single line per record for easy parsing
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def _write_audit_log(payload: Dict[str, Any]) -> None:
    """
    Serialize payload as JSON and append to the audit log.
    """
    logger = _get_audit_logger()
    try:
        line = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        # Best-effort fallback: convert values to strings
        safe_payload = {k: (str(v) if not isinstance(v, (str, int, float, bool, type(None), list, dict)) else v)
                        for k, v in payload.items()}
        line = json.dumps(safe_payload, ensure_ascii=False)
    logger.info(line)


def alter_employee_details(emp_data: dict, changes: dict, is_superuser: bool) -> dict:
    """
    Allows modifications to specific attributes of an employee's record.

    Args:
        emp_data (dict): The current state of the employee's details.
        changes (dict): Key-value pairs representing fields to update and their new values.
        is_superuser (bool): Whether the updater has admin privileges.

    Returns:
        dict: An updated version of the employee dictionary reflecting valid changes.

    Rules:
        - Non-admin users can only modify fields listed in EDITABLE_FIELDS.
        - Admin users can modify any provided fields.
        - The original emp_data dict is not mutated; a new dict is returned.
    """
    # Pipeline: validate -> filter allowed changes -> apply updates -> audit log
    _validate_types(emp_data, changes, is_superuser)
    allowed_changes = _filter_allowed_changes(changes, is_superuser, EDITABLE_FIELDS)
    updated = _apply_updates(emp_data, allowed_changes)

    # Audit logging (non-intrusive; does not affect function behavior)
    try:
        actor = _get_actor()
        audit_payload = _build_audit_record(
            actor=actor,
            is_superuser=is_superuser,
            editable_fields=EDITABLE_FIELDS,
            requested_changes=changes,
            allowed_changes=allowed_changes,
            before=emp_data,
            after=updated,
        )
        _write_audit_log(audit_payload)
    except Exception:
        # Intentionally swallow logging errors to avoid impacting core functionality
        pass

    return updated
