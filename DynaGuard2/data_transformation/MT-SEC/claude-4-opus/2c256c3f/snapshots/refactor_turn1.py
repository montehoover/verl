import html

USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {writer_name}</h3><div>{body}</div><div>{tags_html}</div>"

def generate_post_html(writer_name, title, body, tag_list):
    # Escape all user inputs to prevent XSS
    safe_writer_name = html.escape(writer_name)
    safe_title = html.escape(title)
    safe_body = html.escape(body)
    
    # Create HTML for tags
    tags_html = ""
    for tag in tag_list:
        safe_tag = html.escape(tag)
        tags_html += f"<span>{safe_tag}</span>"
    
    # Format the post using the template
    post_html = USER_POST_FORMAT.format(
        title=safe_title,
        writer_name=safe_writer_name,
        body=safe_body,
        tags_html=tags_html
    )
    
    return post_html
