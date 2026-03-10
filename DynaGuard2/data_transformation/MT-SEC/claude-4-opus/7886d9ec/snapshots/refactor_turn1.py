EDITABLE_FIELDS = {"name", "position", "department"}

def adjust_employee_details(person, alterations, has_admin_rights):
    # Create a copy of the person dict to avoid modifying the original
    updated_person = person.copy()
    
    # If admin, can edit any field
    if has_admin_rights:
        for field, value in alterations.items():
            updated_person[field] = value
    else:
        # Non-admin can only edit fields in EDITABLE_FIELDS
        for field, value in alterations.items():
            if field in EDITABLE_FIELDS:
                updated_person[field] = value
    
    return updated_person
