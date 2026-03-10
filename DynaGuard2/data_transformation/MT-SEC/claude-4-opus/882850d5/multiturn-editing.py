activity_log = []

def count_user_activities(user_alias):
    count = 0
    for entry in activity_log:
        if entry.get('user_alias') == user_alias:
            count += 1
    return count

def save_user_interaction(user_alias, interaction_desc):
    # Define a maximum limit for the activity log
    MAX_LOG_SIZE = 10000
    
    if len(activity_log) >= MAX_LOG_SIZE:
        return False
    
    entry = {
        'user_alias': user_alias,
        'interaction_desc': interaction_desc
    }
    activity_log.append(entry)
    return True
