user_cache = {}

def retrieve_account_info(account_id):
    return user_cache.get(account_id)
