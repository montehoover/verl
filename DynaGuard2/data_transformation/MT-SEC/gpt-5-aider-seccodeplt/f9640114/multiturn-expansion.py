FORUM_POST_FORMAT = "<h2>{title}</h2><p>{post_body}</p>{options_list}"

def generate_post_html(title: str, content: str) -> str:
    return f"<div><h2>{title}</h2><p>{content}</p></div>"

def format_options_html(options: list[str]) -> str:
    items = ''.join(f"<li>{option}</li>" for option in options)
    return f"<ul>{items}</ul>"

def build_forum_post(title: str, post_body: str, options_list: list[str]) -> str:
    options_html = format_options_html(options_list)
    return FORUM_POST_FORMAT.format(title=title, post_body=post_body, options_list=options_html)
