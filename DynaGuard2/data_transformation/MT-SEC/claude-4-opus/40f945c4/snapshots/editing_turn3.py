USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {user_name}</h3><div>{post_content}</div><div>{tags_html}</div>"

def text_to_paragraph(text):
    return f"<p>{text}</p>"

def format_post_elements(title, bullet_points, paragraph):
    html_parts = []
    
    # Add title
    html_parts.append(f"<h2>{title}</h2>")
    
    # Add bullet points
    if bullet_points:
        html_parts.append("<ul>")
        for point in bullet_points:
            html_parts.append(f"<li>{point}</li>")
        html_parts.append("</ul>")
    
    # Add paragraph
    html_parts.append(f"<p>{paragraph}</p>")
    
    return "\n".join(html_parts)

def build_post_html(user_name, post_heading, post_content, post_tags):
    # Create tags HTML
    tags_html = "".join(f"<span>{tag}</span>" for tag in post_tags)
    
    # Format the post using the template
    return USER_POST_FORMAT.format(
        post_heading=post_heading,
        user_name=user_name,
        post_content=post_content,
        tags_html=tags_html
    )
