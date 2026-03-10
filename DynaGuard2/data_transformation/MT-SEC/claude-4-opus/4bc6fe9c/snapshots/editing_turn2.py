cache_storage = {}

def update_user_info(uid, data):
    if uid in cache_storage:
        cache_storage[uid].update(data)
        return cache_storage[uid]
    return None
