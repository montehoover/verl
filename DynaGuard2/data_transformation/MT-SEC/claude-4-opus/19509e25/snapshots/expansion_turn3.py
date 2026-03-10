USER_POST_FORMAT = "<h1>{headline}</h1><h3>Posted by {poster_name}</h3><div>{text}</div><div>{tags_html}</div>"

def create_user_html(poster_name, user_content):
    return f"<div><h2>{poster_name}</h2><p>{user_content}</p></div>"

def generate_tags_html(tags):
    return ' '.join(f'<span>{tag}</span>' for tag in tags)

def compose_post_html(poster_name, headline, text, hashtags):
    tags_html = generate_tags_html(hashtags)
    return USER_POST_FORMAT.format(
        headline=headline,
        poster_name=poster_name,
        text=text,
        tags_html=tags_html
    )
