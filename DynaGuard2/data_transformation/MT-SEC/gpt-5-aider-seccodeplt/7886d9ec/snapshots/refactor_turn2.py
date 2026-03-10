EDITABLE_FIELDS = {"name", "position", "department"}


def _require_dict(name: str, value) -> None:
    """
    Ensure the provided value is a dictionary.

    Raises:
        TypeError: If the value is not a dict.
    """
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dict")


def can_edit_field(field: str, has_admin_rights: bool) -> bool:
    """
    Determine whether a given field can be edited based on permission level.

    - Admins: may edit any field.
    - Non-admins: may edit only fields listed in EDITABLE_FIELDS.
    """
    return has_admin_rights or field in EDITABLE_FIELDS


def filter_permitted_alterations(alterations: dict, has_admin_rights: bool) -> dict:
    """
    Return a new dict containing only those alterations that are permitted
    for the given permission level.
    """
    if has_admin_rights:
        # Admin can apply all alterations
        return dict(alterations)
    # Non-admin: filter to allowed fields only
    return {k: v for k, v in alterations.items() if can_edit_field(k, has_admin_rights)}


def apply_updates(person: dict, updates: dict) -> dict:
    """
    Apply updates to a copy of the person dict and return the updated copy.
    This function does not mutate the input dictionaries.
    """
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
