EDITABLE_FIELDS = {"name", "position", "department"}

def adjust_employee_details(person, alterations, has_admin_rights):
    """
    Modifies certain fields within an employee's profile.
    
    Args:
        person (dict): The current employee profile dictionary
        alterations (dict): Fields and new values to update
        has_admin_rights (bool): Whether the user has admin rights
        
    Returns:
        dict: The revised employee profile after updates
    """
    # Create a copy of the person dict to avoid modifying the original
    updated_person = person.copy()
    
    # Process each alteration
    for field, new_value in alterations.items():
        # If user has admin rights, they can modify any field
        if has_admin_rights:
            updated_person[field] = new_value
        # Non-admin users can only modify fields in EDITABLE_FIELDS
        elif field in EDITABLE_FIELDS:
            updated_person[field] = new_value
        # Skip fields that non-admin users cannot edit
        
    return updated_person
