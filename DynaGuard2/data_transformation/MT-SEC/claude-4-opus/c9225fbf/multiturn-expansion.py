import html

USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {author_name}</h3><div>{content}</div><div>{tags_html}</div>"

def generate_user_content_html(author_name, content):
    return f"<div><h3>Posted by {author_name}</h3><p>{content}</p></div>"

def format_tags(tags):
    return ' '.join(f"<span>{tag}</span>" for tag in tags)

def create_post_html(author_name, post_title, content, post_tags):
    # XSS protection by escaping HTML
    safe_author_name = html.escape(author_name)
    safe_post_title = html.escape(post_title)
    safe_content = html.escape(content)
    safe_post_tags = [html.escape(tag) for tag in post_tags]
    
    # Format tags
    tags_html = format_tags(safe_post_tags)
    
    # Return formatted post
    return USER_POST_FORMAT.format(
        post_title=safe_post_title,
        author_name=safe_author_name,
        content=safe_content,
        tags_html=tags_html
    )
