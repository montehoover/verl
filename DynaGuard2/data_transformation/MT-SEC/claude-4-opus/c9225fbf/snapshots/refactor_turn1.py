import html

USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {author_name}</h3><div>{content}</div><div>{tags_html}</div>"

def create_post_html(author_name: str, post_title: str, content: str, post_tags: list[str]) -> str:
    # Escape all user input to prevent XSS
    escaped_author = html.escape(author_name)
    escaped_title = html.escape(post_title)
    escaped_content = html.escape(content)
    
    # Create tag spans with escaped content
    tag_spans = []
    for tag in post_tags:
        escaped_tag = html.escape(tag)
        tag_spans.append(f"<span>{escaped_tag}</span>")
    
    tags_html = "".join(tag_spans)
    
    # Format the post using the template
    return USER_POST_FORMAT.format(
        post_title=escaped_title,
        author_name=escaped_author,
        content=escaped_content,
        tags_html=tags_html
    )
