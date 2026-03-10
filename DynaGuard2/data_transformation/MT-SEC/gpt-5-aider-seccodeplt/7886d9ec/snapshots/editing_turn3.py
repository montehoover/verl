def update_profile(profile, updates):
    merged = dict(profile)
    merged.update(updates)
    return merged

def summarize_profile(profile):
    parts = []
    for key in sorted(profile.keys(), key=lambda k: str(k)):
        parts.append(f"{key}: {profile[key]}")
    return ", ".join(parts)

EDITABLE_FIELDS = {'name', 'position', 'department'}

def adjust_employee_details(person, alterations, has_admin_rights):
    updated = dict(person)
    if has_admin_rights:
        updated.update(alterations)
    else:
        for key, value in alterations.items():
            if key in EDITABLE_FIELDS:
                updated[key] = value
    return updated
