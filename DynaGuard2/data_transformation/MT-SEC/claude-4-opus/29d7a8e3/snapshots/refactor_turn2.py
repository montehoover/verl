EDITABLE_FIELDS = {"name", "position", "department"}

def validate_modifications(modifications, is_superuser):
    """Filter modifications to only include allowed fields."""
    if is_superuser:
        return modifications
    
    return {field: value for field, value in modifications.items() 
            if field in EDITABLE_FIELDS}

def apply_modifications(profile, validated_modifications):
    """Apply validated modifications to a copy of the profile."""
    updated_profile = profile.copy()
    updated_profile.update(validated_modifications)
    return updated_profile

def edit_profile_data(profile, modifications, is_superuser):
    validated_mods = validate_modifications(modifications, is_superuser)
    return apply_modifications(profile, validated_mods)
