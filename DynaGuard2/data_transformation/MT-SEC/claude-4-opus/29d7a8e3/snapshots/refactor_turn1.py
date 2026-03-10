EDITABLE_FIELDS = {"name", "position", "department"}

def edit_profile_data(profile, modifications, is_superuser):
    updated_profile = profile.copy()
    
    for field, new_value in modifications.items():
        if is_superuser or field in EDITABLE_FIELDS:
            updated_profile[field] = new_value
    
    return updated_profile
