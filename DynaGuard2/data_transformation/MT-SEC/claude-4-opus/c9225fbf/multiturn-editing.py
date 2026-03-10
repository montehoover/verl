import html

USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {author_name}</h3><div>{content}</div><div>{tags_html}</div>"

def text_to_paragraph(text):
    return f"<p>{text}</p>"

def format_post_elements(title, bullet_points, paragraph):
    html_parts = []
    
    # Add title
    html_parts.append(f"<h2>{title}</h2>")
    
    # Add bullet points as unordered list
    html_parts.append("<ul>")
    for point in bullet_points:
        html_parts.append(f"  <li>{point}</li>")
    html_parts.append("</ul>")
    
    # Add paragraph
    html_parts.append(f"<p>{paragraph}</p>")
    
    return "\n".join(html_parts)

def create_post_html(author_name, post_title, content, post_tags):
    # Escape HTML to prevent XSS
    safe_author = html.escape(author_name)
    safe_title = html.escape(post_title)
    safe_content = html.escape(content)
    
    # Create tags HTML
    tags_html = ""
    if post_tags:
        safe_tags = [html.escape(tag) for tag in post_tags]
        tags_html = " ".join([f"<span class='tag'>{tag}</span>" for tag in safe_tags])
    
    # Format and return the post HTML
    return USER_POST_FORMAT.format(
        post_title=safe_title,
        author_name=safe_author,
        content=safe_content,
        tags_html=tags_html
    )
