PROFILE_FORMAT = "User: {username}<br>Bio: {biography}<br>Status: {status}"

def create_user_info_html(username, user_info):
    return f"<div><h2>{username}</h2><p>{user_info}</p></div>"

def sanitize_user_input(input_string):
    return input_string.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def generate_user_profile(username, biography, status):
    return PROFILE_FORMAT.format(username=username, biography=biography, status=status)
