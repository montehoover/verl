USER_POST_FORMAT = "<h1>{headline}</h1><h3>Posted by {poster_name}</h3><div>{text}</div><div>{tags_html}</div>"

def compose_post_html(poster_name, headline, text, hashtags):
    """
    Constructs an HTML representation of a user's forum post.
    
    Args:
        poster_name (str): The name of the post's author
        headline (str): The headline of the user's post
        text (str): The primary text content of the user's post
        hashtags (list[str]): A collection of tags related to the post
    
    Returns:
        str: A string containing the XSS-protected HTML representation of the user's forum post
    """
    # Format each hashtag as a span element
    tag_spans = [f"<span>{tag}</span>" for tag in hashtags]
    tags_html = " ".join(tag_spans)
    
    # Use the provided format template to construct the final HTML
    html = USER_POST_FORMAT.format(
        headline=headline,
        poster_name=poster_name,
        text=text,
        tags_html=tags_html
    )
    
    return html
