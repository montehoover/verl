def create_user_html(poster_name, user_content):
    return f"<div><h2>{poster_name}</h2><p>{user_content}</p></div>"

def generate_tags_html(tags):
    return ' '.join(f'<span>{tag}</span>' for tag in tags)
