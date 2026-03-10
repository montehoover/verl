import html

FORUM_POST_FORMAT = "<h2>{title}</h2><p>{post_body}</p>{options_list}"

def build_forum_post(title: str, post_body: str, options_list: list[str]) -> str:
    safe_title = html.escape(title, quote=True)
    safe_body = html.escape(post_body, quote=True)

    options_html = ""
    if options_list:
        items_html = "".join(f"<li>{html.escape(str(item), quote=True)}</li>" for item in options_list)
        options_html = f"<ul>{items_html}</ul>"

    return FORUM_POST_FORMAT.format(title=safe_title, post_body=safe_body, options_list=options_html)
