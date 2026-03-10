user_cache = {}

def cache_application_data(client_id: str, info_data: dict) -> bool:
    try:
        user_cache[client_id] = info_data
        return True
    except Exception:
        return False
