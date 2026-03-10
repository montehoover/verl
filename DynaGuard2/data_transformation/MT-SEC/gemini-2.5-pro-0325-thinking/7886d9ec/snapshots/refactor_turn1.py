EDITABLE_FIELDS = {"name", "position", "department"}

def adjust_employee_details(person: dict, alterations: dict, has_admin_rights: bool) -> dict:
    """
    Modifies certain fields within an employee's profile.

    Args:
        person: dict, represents the current employee profile.
        alterations: dict, specifies the fields and new values to update.
        has_admin_rights: bool, indicates if the operation is by an admin.

    Returns:
        dict: The revised employee profile dictionary.
    """
    updated_person = person.copy()  # Work on a copy to avoid modifying the original dict directly

    for key, value in alterations.items():
        if has_admin_rights:
            updated_person[key] = value
        elif key in EDITABLE_FIELDS:
            updated_person[key] = value
            
    return updated_person
