EDITABLE_FIELDS = {'name', 'position', 'department'}

def update_dict_values(data, updates):
    return {**data, **updates}

def edit_profile_data(profile, modifications, is_superuser):
    if is_superuser:
        return {**profile, **modifications}
    else:
        filtered_modifications = {k: v for k, v in modifications.items() if k in EDITABLE_FIELDS}
        return {**profile, **filtered_modifications}
