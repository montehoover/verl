import logging

EDITABLE_FIELDS = {"name", "position", "department"}
SENSITIVE_FIELDS = {"salary", "social_security_number"}

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _require_dict(name: str, value) -> None:
    """
    Ensure the provided value is a dictionary.

    Raises:
        TypeError: If the value is not a dict.
    """
    if isinstance(value, dict):
        return
    raise TypeError(f"{name} must be a dict")


def can_edit_field(field: str, has_admin_rights: bool) -> bool:
    """
    Determine whether a given field can be edited based on permission level.

    - Admins: may edit any field.
    - Non-admins: may edit only fields listed in EDITABLE_FIELDS.
    """
    return has_admin_rights or field in EDITABLE_FIELDS


def _redact_for_logging(field: str, value):
    """
    Redact sensitive fields for logging output.
    """
    if field in SENSITIVE_FIELDS:
        return "<redacted>"
    return value


def filter_permitted_alterations(alterations: dict, has_admin_rights: bool) -> dict:
    """
    Return a new dict containing only those alterations that are permitted
    for the given permission level.
    """
    if not alterations:
        logger.debug("No alterations provided; nothing to filter.")
        return {}

    if has_admin_rights:
        logger.debug(
            "Admin rights granted; permitting all alterations: %s",
            ", ".join(sorted(alterations.keys())),
        )
        return dict(alterations)

    # Non-admin: filter to allowed fields only
    permitted = {k: v for k, v in alterations.items() if can_edit_field(k, has_admin_rights)}
    disallowed = sorted(set(alterations.keys()) - set(permitted.keys()))
    if disallowed:
        logger.debug("Non-admin: ignoring disallowed fields: %s", ", ".join(disallowed))
    if permitted:
        logger.debug("Non-admin: permitted fields: %s", ", ".join(sorted(permitted.keys())))
    else:
        logger.debug("Non-admin: no permitted alterations found.")
    return permitted


def _summarize_changes(person: dict, updates: dict) -> dict:
    """
    Build a dict of effective changes: field -> (old_value, new_value),
    excluding fields where the value is unchanged.
    """
    changes = {}
    for field, new_val in updates.items():
        old_val = person.get(field)
        if old_val != new_val:
            changes[field] = (old_val, new_val)
    return changes


def apply_updates(person: dict, updates: dict) -> dict:
    """
    Apply updates to a copy of the person dict and return the updated copy.
    This function does not mutate the input dictionaries.
    """
    changes = _summarize_changes(person, updates)
    if changes:
        parts = []
        for field, (old_val, new_val) in sorted(changes.items()):
            parts.append(
                f"{field}: {repr(_redact_for_logging(field, old_val))} -> {repr(_redact_for_logging(field, new_val))}"
            )
        logger.info("Applying %d update(s) to employee profile: %s", len(changes), "; ".join(parts))
    else:
        logger.info("No effective changes to apply to employee profile.")

    updated = person.copy()
    updated.update(updates)
    return updated


def adjust_employee_details(person: dict, alterations: dict, has_admin_rights: bool) -> dict:
    """
    Modify an employee's profile based on provided alterations and access level.

    - Non-admin users: may only update fields in EDITABLE_FIELDS.
    - Admin users: may update any field (including adding new ones).

    Returns a new dict with the updates applied; does not mutate the input dict.
    """
    _require_dict("person", person)
    _require_dict("alterations", alterations)

    permitted_updates = filter_permitted_alterations(alterations, has_admin_rights)
    return apply_updates(person, permitted_updates)
