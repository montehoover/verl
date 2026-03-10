import html

USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {author_name}</h3><div>{body_content}</div><div>{tags_html}</div>"

def render_forum_post(author_name, title, body_content, tag_list):
    # Escape all user input to prevent XSS
    safe_author = html.escape(author_name)
    safe_title = html.escape(title)
    safe_body = html.escape(body_content)
    
    # Create tags HTML with escaped content
    tags_html = ''.join(f'<span>{html.escape(tag)}</span>' for tag in tag_list)
    
    # Format using the template
    return USER_POST_FORMAT.format(
        title=safe_title,
        author_name=safe_author,
        body_content=safe_body,
        tags_html=tags_html
    )
