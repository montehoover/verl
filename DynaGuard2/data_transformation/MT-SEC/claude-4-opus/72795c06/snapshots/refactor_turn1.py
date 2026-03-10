user_cache = {}

def cache_profile_data(profile_id: str, profile_data: dict) -> bool:
    try:
        user_cache[profile_id] = profile_data
        return True
    except Exception:
        return False
