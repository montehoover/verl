EDITABLE_FIELDS = {'name', 'position', 'department'}

def update_profile(profile, updates):
    profile.update(updates)
    return profile

def summarize_profile(profile):
    sorted_items = sorted(profile.items())
    return '\n'.join(f'{key}: {value}' for key, value in sorted_items)

def adjust_employee_details(person, alterations, has_admin_rights):
    if has_admin_rights:
        person.update(alterations)
    else:
        for field, value in alterations.items():
            if field in EDITABLE_FIELDS:
                person[field] = value
    return person
