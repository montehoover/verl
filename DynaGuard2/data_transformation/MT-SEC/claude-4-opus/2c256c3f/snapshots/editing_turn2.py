import html

def basic_post_html(author, post_title):
    return f"<h1>{post_title}</h1><h3>Posted by {author}</h3>"

def detailed_post_html(author, post_title, content):
    safe_author = html.escape(author)
    safe_title = html.escape(post_title)
    safe_content = html.escape(content)
    return f"<h1>{safe_title}</h1><h3>Posted by {safe_author}</h3><div>{safe_content}</div>"
