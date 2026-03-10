EDITABLE_FIELDS = {'name', 'position', 'department'}


def update_dict_values(data, updates):
    """
    Merge two dictionaries by applying the key-value pairs from `updates` to `data`.
    Returns a new dictionary and does not mutate the inputs.
    """
    if not isinstance(data, dict) or not isinstance(updates, dict):
        raise TypeError("Both 'data' and 'updates' must be dictionaries")
    result = data.copy()
    result.update(updates)
    return result


def edit_profile_data(profile, modifications, is_superuser):
    """
    Update an employee profile based on permissions.

    - Non-superusers can only edit fields in EDITABLE_FIELDS.
    - Superusers can edit any field.
    Returns a new profile dictionary and does not mutate the input.
    """
    if not isinstance(profile, dict) or not isinstance(modifications, dict):
        raise TypeError("'profile' and 'modifications' must be dictionaries")
    if not isinstance(is_superuser, bool):
        raise TypeError("'is_superuser' must be a boolean")

    updated = profile.copy()

    if is_superuser:
        updated.update(modifications)
        return updated

    for key, value in modifications.items():
        if key in EDITABLE_FIELDS:
            updated[key] = value
    return updated
