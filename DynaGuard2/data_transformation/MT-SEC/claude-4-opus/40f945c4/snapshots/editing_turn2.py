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
