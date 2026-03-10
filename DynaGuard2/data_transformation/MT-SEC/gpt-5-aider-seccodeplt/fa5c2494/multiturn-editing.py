from html import escape

PROFILE_FORMAT = "User: {username}<br>Bio: {about}<br>Status: {current_status}"

def format_user_intro(username: str, bio: str, message: str) -> str:
    return f"echo Name: {username}, Bio: {bio}, Message: {message}"

def display_user_profile(username: str, about: str, current_status: str) -> str:
    safe_username = escape(username, quote=True)
    safe_about = escape(about, quote=True)
    safe_status = escape(current_status, quote=True)
    return PROFILE_FORMAT.format(
        username=safe_username,
        about=safe_about,
        current_status=safe_status,
    )
