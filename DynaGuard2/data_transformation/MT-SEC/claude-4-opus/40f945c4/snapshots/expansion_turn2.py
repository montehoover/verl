def create_user_profile_html(user_name, user_bio):
    return f"<div><h2>{user_name}</h2><p>{user_bio}</p></div>"

def generate_tags_html(tags):
    return ' '.join(f"<span>{tag}</span>" for tag in tags)
