import html

USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {writer_name}</h3><div>{body}</div><div>{tags_html}</div>"

def generate_tags_html(tag_list):
    """Generate HTML for tags, with each tag in a separate span element."""
    return ''.join(f"<span>{html.escape(tag)}</span>" for tag in tag_list)

def sanitize_inputs(writer_name, title, body):
    """Sanitize all text inputs to prevent XSS attacks."""
    return {
        'writer_name': html.escape(writer_name),
        'title': html.escape(title),
        'body': html.escape(body)
    }

def generate_post_html(writer_name, title, body, tag_list):
    # Step 1: Sanitize all text inputs
    sanitized = sanitize_inputs(writer_name, title, body)
    
    # Step 2: Generate tags HTML
    tags_html = generate_tags_html(tag_list)
    
    # Step 3: Format the post using the template
    post_html = USER_POST_FORMAT.format(
        title=sanitized['title'],
        writer_name=sanitized['writer_name'],
        body=sanitized['body'],
        tags_html=tags_html
    )
    
    return post_html
