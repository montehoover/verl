cache_storage = {}

def get_user_info(uid):
    return cache_storage.get(uid)
