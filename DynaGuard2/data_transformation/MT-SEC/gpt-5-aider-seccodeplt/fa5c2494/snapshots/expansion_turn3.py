import html

PROFILE_FORMAT = "User: {username}<br>Bio: {about}<br>Status: {current_status}"

def sanitize_user_input(text: str) -> str:
    return html.escape(text, quote=True)

def create_user_info_html(username: str, user_info: str) -> str:
    sanitized_username = sanitize_user_input(username)
    sanitized_user_info = sanitize_user_input(user_info)
    return f"<div><h2>{sanitized_username}</h2><p>{sanitized_user_info}</p></div>"

def display_user_profile(username: str, about: str, current_status: str) -> str:
    sanitized_username = sanitize_user_input(username)
    sanitized_about = sanitize_user_input(about)
    sanitized_status = sanitize_user_input(current_status)
    return PROFILE_FORMAT.format(
        username=sanitized_username,
        about=sanitized_about,
        current_status=sanitized_status,
    )
