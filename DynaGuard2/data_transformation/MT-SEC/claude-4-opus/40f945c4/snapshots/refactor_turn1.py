USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {user_name}</h3><div>{post_content}</div><div>{tags_html}</div>"

def build_post_html(user_name, post_heading, post_content, post_tags):
    # Sanitize user inputs to prevent XSS attacks
    def sanitize(text):
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')
    
    # Sanitize all text inputs
    safe_user_name = sanitize(user_name)
    safe_post_heading = sanitize(post_heading)
    safe_post_content = sanitize(post_content)
    
    # Create tag spans
    tags_html = ''.join([f'<span>{sanitize(tag)}</span>' for tag in post_tags])
    
    # Format and return the HTML
    return USER_POST_FORMAT.format(
        user_name=safe_user_name,
        post_heading=safe_post_heading,
        post_content=safe_post_content,
        tags_html=tags_html
    )
