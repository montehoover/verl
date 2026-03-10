import html

USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {author_name}</h3><div>{body_content}</div><div>{tags_html}</div>"

def render_forum_post(author_name, title, body_content, tag_list):
    # Escape all user input to prevent XSS
    safe_author = html.escape(author_name)
    safe_title = html.escape(title)
    safe_body = html.escape(body_content)
    
    # Create escaped tag spans
    tag_spans = []
    for tag in tag_list:
        safe_tag = html.escape(tag)
        tag_spans.append(f"<span>{safe_tag}</span>")
    
    tags_html = "".join(tag_spans)
    
    # Format the post using the template
    return USER_POST_FORMAT.format(
        title=safe_title,
        author_name=safe_author,
        body_content=safe_body,
        tags_html=tags_html
    )
