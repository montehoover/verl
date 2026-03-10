activity_log = []

def get_user_activities(user_alias):
    user_activities = []
    for entry in activity_log:
        if entry.get('user_alias') == user_alias:
            user_activities.append(entry.get('interaction_desc'))
    return user_activities
