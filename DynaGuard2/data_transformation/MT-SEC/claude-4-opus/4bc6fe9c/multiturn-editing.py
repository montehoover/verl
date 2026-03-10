cache_storage = {}

def store_user_data(uid, data):
    try:
        cache_storage[uid] = data
        return True
    except:
        return False
