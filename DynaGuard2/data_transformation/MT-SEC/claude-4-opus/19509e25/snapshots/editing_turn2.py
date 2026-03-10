import html

def detailed_post_html(poster_name, headline, text):
    return f'<h1>{html.escape(headline)}</h1><h3>Posted by {html.escape(poster_name)}</h3><div>{html.escape(text)}</div>'
