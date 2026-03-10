import logging
from typing import Dict, Any, Optional, Iterable

EDITABLE_FIELDS = {"name", "position", "department"}

# Module-level logger configured for human-readable auditing
logger = logging.getLogger("profile_edit")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Optional global actor for audit logs (who performed the action)
_CURRENT_AUDIT_ACTOR: Optional[str] = None


def set_audit_actor(actor: Optional[str]) -> None:
    """
    Set the current actor (username, email, or user id) for audit logs.
    Pass None to clear and revert to 'unknown'.
    """
    global _CURRENT_AUDIT_ACTOR
    _CURRENT_AUDIT_ACTOR = actor


def _get_audit_actor() -> str:
    return _CURRENT_AUDIT_ACTOR or "unknown"


def _format_fields(fields: Iterable[str]) -> str:
    items = list(fields)
    return "none" if not items else ", ".join(sorted(items))


def validate_modifications(modifications: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize and validate the modifications input.

    Returns:
        A dictionary of modifications (empty dict if None provided).
    """
    if modifications is None:
        return {}
    if not isinstance(modifications, dict):
        raise ValueError("modifications must be a dict or None")
    return dict(modifications)


def select_allowed_edits(modifications: Dict[str, Any], is_superuser: bool, profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Select which modifications are allowed based on privilege level and existing profile keys.
    Admins can edit any existing key in the profile.
    Non-admins can only edit keys in EDITABLE_FIELDS that also exist in the profile.
    """
    profile_keys = set(profile.keys())

    if is_superuser:
        return {k: v for k, v in modifications.items() if k in profile_keys}

    allowed_keys = EDITABLE_FIELDS & profile_keys
    return {k: v for k, v in modifications.items() if k in allowed_keys}


def apply_edited_fields(profile: Dict[str, Any], edits: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply allowed edits to a copy of the profile and return the updated profile.
    """
    if not edits:
        return dict(profile)

    updated = dict(profile)
    updated.update(edits)
    return updated


def _profile_identifier(profile: Dict[str, Any]) -> str:
    """
    Choose a human-friendly identifier for the profile for logging.
    Preference order: id, employee_id, name, or a placeholder.
    """
    for key in ("id", "employee_id", "name"):
        if key in profile and profile[key]:
            return str(profile[key])
    return "<unknown profile>"


def _log_edit_attempt(profile: Dict[str, Any], modifications: Dict[str, Any], is_superuser: bool) -> None:
    actor = _get_audit_actor()
    user_type = "admin" if is_superuser else "non-admin"
    profile_id = _profile_identifier(profile)
    attempted_fields = _format_fields(modifications.keys())
    logger.info(
        f"Edit attempt by {actor} ({user_type}) on profile {profile_id} "
        f"- fields attempted: {attempted_fields}"
    )


def _log_edit_result(profile: Dict[str, Any], allowed_edits: Dict[str, Any], blocked_fields: Iterable[str], is_superuser: bool) -> None:
    actor = _get_audit_actor()
    user_type = "admin" if is_superuser else "non-admin"
    profile_id = _profile_identifier(profile)

    changed_fields = _format_fields(allowed_edits.keys())
    blocked_str = _format_fields(blocked_fields)

    if allowed_edits:
        logger.info(
            f"Edit applied by {actor} ({user_type}) on profile {profile_id} "
            f"- fields changed: {changed_fields}"
        )
    else:
        logger.info(
            f"No changes applied by {actor} ({user_type}) on profile {profile_id}"
        )

    if blocked_str != "none":
        logger.info(f"Fields blocked from editing: {blocked_str}")


def edit_profile_data(profile: Dict[str, Any], modifications: Optional[Dict[str, Any]], is_superuser: bool) -> Dict[str, Any]:
    """
    Modify an employee profile based on the provided modifications and privilege level.

    Args:
        profile: Current employee profile dictionary.
        modifications: Dictionary of fields and new values to apply.
        is_superuser: Whether the caller has admin privileges.

    Returns:
        A new dictionary representing the updated employee profile.
    """
    if not isinstance(profile, dict):
        raise ValueError("profile must be a dict")

    normalized_mods = validate_modifications(modifications)

    # Log the attempt before enforcing permissions
    _log_edit_attempt(profile, normalized_mods, is_superuser)

    # Guard: nothing to do if there are no modifications
    if not normalized_mods:
        _log_edit_result(profile, {}, [], is_superuser)
        return dict(profile)

    allowed_edits = select_allowed_edits(normalized_mods, is_superuser, profile)
    blocked_fields = set(normalized_mods.keys()) - set(allowed_edits.keys())

    # Log what was actually applied and what was blocked
    _log_edit_result(profile, allowed_edits, blocked_fields, is_superuser)

    return apply_edited_fields(profile, allowed_edits)
