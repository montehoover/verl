activity_log = []

def count_user_activities(user_alias):
    count = 0
    for entry in activity_log:
        if entry.get('user_alias') == user_alias:
            count += 1
    return count
