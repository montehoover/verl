USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {author_name}</h3><div>{body_content}</div><div>{tags_html}</div>"

def create_user_html(author_name, user_bio):
    return f"<div><h2>{author_name}</h2><p>{user_bio}</p></div>"

def generate_tags_html(tags):
    return ' '.join(f"<span>{tag}</span>" for tag in tags)

def render_forum_post(author_name, title, body_content, tag_list):
    tags_html = generate_tags_html(tag_list)
    return USER_POST_FORMAT.format(
        title=title,
        author_name=author_name,
        body_content=body_content,
        tags_html=tags_html
    )
