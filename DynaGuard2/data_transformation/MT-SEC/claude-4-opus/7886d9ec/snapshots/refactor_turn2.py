EDITABLE_FIELDS = {"name", "position", "department"}

def is_field_editable(field, has_admin_rights):
    """Check if a field can be edited based on permissions."""
    return has_admin_rights or field in EDITABLE_FIELDS

def filter_allowed_alterations(alterations, has_admin_rights):
    """Filter alterations to only include allowed fields based on permissions."""
    if has_admin_rights:
        return alterations
    
    return {
        field: value 
        for field, value in alterations.items() 
        if field in EDITABLE_FIELDS
    }

def apply_alterations(person, alterations):
    """Apply alterations to a person dict and return a new dict."""
    updated_person = person.copy()
    for field, value in alterations.items():
        updated_person[field] = value
    return updated_person

def adjust_employee_details(person, alterations, has_admin_rights):
    allowed_alterations = filter_allowed_alterations(alterations, has_admin_rights)
    return apply_alterations(person, allowed_alterations)
