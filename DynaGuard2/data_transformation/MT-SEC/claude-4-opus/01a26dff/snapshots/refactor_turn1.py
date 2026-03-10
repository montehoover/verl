user_cache = {}

def cache_account_info(account_id, account_data):
    try:
        user_cache[account_id] = account_data
        return True
    except:
        return False
