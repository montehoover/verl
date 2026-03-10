def update_profile(profile, updates):
    merged = dict(profile)
    merged.update(updates)
    return merged

def summarize_profile(profile):
    parts = []
    for key in sorted(profile.keys(), key=lambda k: str(k)):
        parts.append(f"{key}: {profile[key]}")
    return ", ".join(parts)
