def create_user_profile_html(username, bio):
    return f"<div><h2>{username}</h2><p>{bio}</p></div>"

def generate_tags_html(tags):
    return ' '.join(f"<span>{tag}</span>" for tag in tags)
