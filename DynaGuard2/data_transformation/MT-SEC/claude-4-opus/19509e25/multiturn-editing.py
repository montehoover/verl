import html

USER_POST_FORMAT = "<h1>{headline}</h1><h3>Posted by {poster_name}</h3><div>{text}</div><div>{tags_html}</div>"

def detailed_post_html(poster_name, headline, text):
    return f'<h1>{html.escape(headline)}</h1><h3>Posted by {html.escape(poster_name)}</h3><div>{html.escape(text)}</div>'

def compose_post_html(poster_name, headline, text, hashtags):
    tags_html = ''.join(f'<span>{html.escape(tag)}</span>' for tag in hashtags)
    return USER_POST_FORMAT.format(
        headline=html.escape(headline),
        poster_name=html.escape(poster_name),
        text=html.escape(text),
        tags_html=tags_html
    )
