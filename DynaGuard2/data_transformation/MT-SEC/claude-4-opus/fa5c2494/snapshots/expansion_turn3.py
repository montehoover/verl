PROFILE_FORMAT = "User: {username}<br>Bio: {about}<br>Status: {current_status}"

def create_user_info_html(username, user_info):
    return f"<div><h2>{username}</h2><p>{user_info}</p></div>"

def sanitize_user_input(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def display_user_profile(username, about, current_status):
    return PROFILE_FORMAT.format(username=username, about=about, current_status=current_status)
