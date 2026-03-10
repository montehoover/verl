EDITABLE_FIELDS = {"name", "position", "department"}

def adjust_employee_details(person: dict, alterations: dict, has_admin_rights: bool) -> dict:
    """
    Modifies certain fields within an employee's profile.

    Args:
      person: dict, represents the current employee profile in the form of a dictionary with their details.
      alterations: dict, a dictionary specifying the fields and respective new values meant to update the employee profile.
      has_admin_rights: bool, indicates whether the operation is carried out by an admin or a non-admin user.

    Returns:
      dict: The revised version of the employee profile dictionary after the appropriate fields have been updated.
    """
    updated_person = person.copy()  # Work on a copy to avoid modifying the original dict directly if it's not desired outside the function

    for field, new_value in alterations.items():
        if has_admin_rights:
            if field in updated_person: # Ensure the field exists before updating, or allow adding new fields if admin
                updated_person[field] = new_value
        elif field in EDITABLE_FIELDS:
            if field in updated_person: # Ensure the field exists before updating
                updated_person[field] = new_value
    
    return updated_person
