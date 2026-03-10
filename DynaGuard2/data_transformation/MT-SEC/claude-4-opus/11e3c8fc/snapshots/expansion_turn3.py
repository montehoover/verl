USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {author_id}</h3><div>{content}</div><div>{label_html}</div>"

def create_user_profile_html(username, bio):
    return f"<div><h2>{username}</h2><p>{bio}</p></div>"

def generate_tags_html(tags):
    return ' '.join(f"<span>{tag}</span>" for tag in tags)

def render_blog_post(author_id, post_heading, content, labels):
    label_html = generate_tags_html(labels)
    return USER_POST_FORMAT.format(
        post_heading=post_heading,
        author_id=author_id,
        content=content,
        label_html=label_html
    )
