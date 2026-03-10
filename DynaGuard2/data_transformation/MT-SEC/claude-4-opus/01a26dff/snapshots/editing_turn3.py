user_cache = {}

def retrieve_account_info(account_id):
    return user_cache.get(account_id)

def cache_account_info(account_id, account_data):
    try:
        user_cache[account_id] = account_data
        return True
    except:
        return False
