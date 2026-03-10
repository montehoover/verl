PROFILE_FORMAT = "User: {username}<br>Bio: {biography}<br>Status: {status}"

def display_user_info(username: str, biography: str) -> str:
    return f"User: {username}, Bio: {biography}"

def generate_user_profile(username: str, biography: str, status: str) -> str:
    return PROFILE_FORMAT.format(username=username, biography=biography, status=status)
