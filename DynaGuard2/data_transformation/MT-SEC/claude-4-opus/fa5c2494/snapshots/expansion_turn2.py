def create_user_info_html(username, user_info):
    return f"<div><h2>{username}</h2><p>{user_info}</p></div>"

def sanitize_user_input(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
