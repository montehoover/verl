FORUM_POST_FORMAT = "<h2>{title}</h2><p>{post_body}</p>{options_list}"

def generate_post_html(title, content):
    return f"<div><h2>{title}</h2><p>{content}</p></div>"

def format_options_html(options):
    li_elements = ''.join(f'<li>{option}</li>' for option in options)
    return f'<ul>{li_elements}</ul>'

def build_forum_post(title, post_body, options_list):
    options_html = format_options_html(options_list)
    return FORUM_POST_FORMAT.format(title=title, post_body=post_body, options_list=options_html)
