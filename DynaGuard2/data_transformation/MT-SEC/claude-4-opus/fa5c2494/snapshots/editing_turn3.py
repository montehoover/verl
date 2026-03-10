PROFILE_FORMAT = "User: {username}<br>Bio: {about}<br>Status: {current_status}"

def display_user_profile(username, about, current_status):
    return PROFILE_FORMAT.format(username=username, about=about, current_status=current_status)
