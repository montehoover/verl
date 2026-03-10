import html

USER_POST_FORMAT = "<h1>{headline}</h1><h3>Posted by {poster_name}</h3><div>{text}</div><div>{tags_html}</div>"

def compose_post_html(poster_name, headline, text, hashtags):
    # Escape all user inputs to prevent XSS
    safe_poster_name = html.escape(poster_name)
    safe_headline = html.escape(headline)
    safe_text = html.escape(text)
    
    # Create HTML for tags, each in its own span
    tags_html = ""
    for tag in hashtags:
        safe_tag = html.escape(tag)
        tags_html += f"<span>{safe_tag}</span>"
    
    # Format and return the complete HTML
    return USER_POST_FORMAT.format(
        headline=safe_headline,
        poster_name=safe_poster_name,
        text=safe_text,
        tags_html=tags_html
    )
