EDITABLE_FIELDS = {"name", "position", "department"}

def _can_edit_field(field_name: str, has_admin_rights: bool) -> bool:
    """Checks if a field can be edited based on user rights."""
    if has_admin_rights:
        return True
    return field_name in EDITABLE_FIELDS

def _apply_alteration(profile: dict, field_name: str, new_value: any) -> None:
    """Applies a single alteration to the profile."""
    profile[field_name] = new_value

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

    for field_name, new_value in alterations.items():
        if _can_edit_field(field_name, has_admin_rights):
            _apply_alteration(updated_person, field_name, new_value)
            
    return updated_person
