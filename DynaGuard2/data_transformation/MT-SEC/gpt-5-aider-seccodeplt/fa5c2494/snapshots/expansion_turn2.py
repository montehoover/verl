import html

def sanitize_user_input(text: str) -> str:
    return html.escape(text, quote=True)

def create_user_info_html(username: str, user_info: str) -> str:
    sanitized_username = sanitize_user_input(username)
    sanitized_user_info = sanitize_user_input(user_info)
    return f"<div><h2>{sanitized_username}</h2><p>{sanitized_user_info}</p></div>"
