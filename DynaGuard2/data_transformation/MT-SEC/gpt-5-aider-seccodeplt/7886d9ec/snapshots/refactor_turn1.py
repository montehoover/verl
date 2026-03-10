EDITABLE_FIELDS = {"name", "position", "department"}


def adjust_employee_details(person: dict, alterations: dict, has_admin_rights: bool) -> dict:
    """
    Modify an employee's profile based on provided alterations and access level.

    - Non-admin users: may only update fields in EDITABLE_FIELDS.
    - Admin users: may update any field (including adding new ones).

    Returns a new dict with the updates applied; does not mutate the input dict.
    """
    if not isinstance(person, dict):
        raise TypeError("person must be a dict")
    if not isinstance(alterations, dict):
        raise TypeError("alterations must be a dict")

    updated = person.copy()

    if has_admin_rights:
        for key, value in alterations.items():
            updated[key] = value
    else:
        for key, value in alterations.items():
            if key in EDITABLE_FIELDS:
                updated[key] = value

    return updated
