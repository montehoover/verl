EDITABLE_FIELDS = {"name", "position", "department"}

def edit_profile_data(profile, modifications, is_superuser):
    """
    Modifies certain fields within an employee's profile.
    
    Args:
        profile (dict): The current employee profile dictionary
        modifications (dict): Fields and new values to update
        is_superuser (bool): Whether the user is an admin
        
    Returns:
        dict: The revised employee profile after updates
    """
    # Create a copy of the profile to avoid modifying the original
    updated_profile = profile.copy()
    
    # Apply modifications based on user permissions
    for field, new_value in modifications.items():
        if is_superuser or field in EDITABLE_FIELDS:
            updated_profile[field] = new_value
    
    return updated_profile
