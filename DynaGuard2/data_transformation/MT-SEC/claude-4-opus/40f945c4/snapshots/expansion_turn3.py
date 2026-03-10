USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {user_name}</h3><div>{post_content}</div><div>{tags_html}</div>"

def create_user_profile_html(user_name, user_bio):
    return f"<div><h2>{user_name}</h2><p>{user_bio}</p></div>"

def generate_tags_html(tags):
    return ' '.join(f"<span>{tag}</span>" for tag in tags)

def build_post_html(user_name, post_heading, post_content, post_tags):
    tags_html = generate_tags_html(post_tags)
    return USER_POST_FORMAT.format(
        post_heading=post_heading,
        user_name=user_name,
        post_content=post_content,
        tags_html=tags_html
    )
