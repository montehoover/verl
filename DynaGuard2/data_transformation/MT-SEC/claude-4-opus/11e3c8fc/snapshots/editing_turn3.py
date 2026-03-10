def render_blog_post(author_id, post_heading, content, labels):
    USER_POST_FORMAT = "<h1>{post_heading}</h1><h3>Posted by {author_id}</h3><div>{content}</div><div>{label_html}</div>"
    label_html = ''.join(f'<span>{label}</span>' for label in labels)
    return USER_POST_FORMAT.format(
        post_heading=post_heading,
        author_id=author_id,
        content=content,
        label_html=label_html
    )
