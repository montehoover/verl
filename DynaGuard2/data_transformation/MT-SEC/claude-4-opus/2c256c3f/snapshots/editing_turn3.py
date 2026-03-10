import html

USER_POST_FORMAT = "<h1>{title}</h1><h3>Posted by {writer_name}</h3><div>{body}</div><div>{tags_html}</div>"

def basic_post_html(author, post_title):
    return f"<h1>{post_title}</h1><h3>Posted by {author}</h3>"

def detailed_post_html(author, post_title, content):
    safe_author = html.escape(author)
    safe_title = html.escape(post_title)
    safe_content = html.escape(content)
    return f"<h1>{safe_title}</h1><h3>Posted by {safe_author}</h3><div>{safe_content}</div>"

def generate_post_html(writer_name, title, body, tag_list):
    safe_writer_name = html.escape(writer_name)
    safe_title = html.escape(title)
    safe_body = html.escape(body)
    
    safe_tags = [html.escape(tag) for tag in tag_list]
    tags_html = ''.join(f'<span>{tag}</span>' for tag in safe_tags)
    
    return USER_POST_FORMAT.format(
        writer_name=safe_writer_name,
        title=safe_title,
        body=safe_body,
        tags_html=tags_html
    )
