USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {username}</h3><div>{post_body}</div><div>{tags_html}</div>"

def create_user_profile_html(username, bio):
    return f"<div><h2>{username}</h2><p>{bio}</p></div>"

def generate_tags_html(tags):
    return ' '.join(f"<span>{tag}</span>" for tag in tags)

def render_user_post(username, post_title, post_body, tags):
    tags_html = generate_tags_html(tags)
    return USER_POST_FORMAT.format(
        post_title=post_title,
        username=username,
        post_body=post_body,
        tags_html=tags_html
    )
