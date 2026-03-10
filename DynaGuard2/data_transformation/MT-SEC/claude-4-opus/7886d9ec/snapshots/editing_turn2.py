def update_profile(profile, updates):
    profile.update(updates)
    return profile

def summarize_profile(profile):
    sorted_items = sorted(profile.items())
    return '\n'.join(f'{key}: {value}' for key, value in sorted_items)
