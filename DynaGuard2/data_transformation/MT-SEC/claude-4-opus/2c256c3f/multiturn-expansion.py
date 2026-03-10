USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {writer_name}</h3><div>{body}</div><div>{tags_html}</div>"

def create_profile_html(writer_name, bio):
    return f"<div><h2>{writer_name}</h2><p>{bio}</p></div>"

def wrap_tags_in_html(tags):
    return ' '.join(f'<span>{tag}</span>' for tag in tags)

def generate_post_html(writer_name, title, body, tag_list):
    tags_html = wrap_tags_in_html(tag_list)
    return USER_POST_FORMAT.format(
        title=title,
        writer_name=writer_name,
        body=body,
        tags_html=tags_html
    )
