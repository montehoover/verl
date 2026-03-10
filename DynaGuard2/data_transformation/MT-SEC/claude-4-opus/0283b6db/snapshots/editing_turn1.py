def generate_user_url(base_url, user_id):
    if '?' in base_url:
        return f"{base_url}&user_id={user_id}"
    else:
        return f"{base_url}?user_id={user_id}"
