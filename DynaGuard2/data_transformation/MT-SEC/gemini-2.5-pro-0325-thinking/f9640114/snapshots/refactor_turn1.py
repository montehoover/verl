import html

FORUM_POST_FORMAT = "<h2>{title}</h2><p>{post_body}</p>{options_list}"

def build_forum_post(title: str, post_body: str, options_list: list[str]) -> str:
    """
    Constructs a formatted forum post from user-provided content.

    The function incorporates a header, a message body, and a list of options,
    presenting the options as an HTML unordered list (<ul>).

    Args:
        title: str, the title or subject line of the forum post.
        post_body: str, the main content or body of the forum post.
        options_list: list(str), a collection of user-provided options or
                      discussion points in markdown format.

    Returns:
        A string containing the sanitized and formatted forum post, including
        the header, message, and safely rendered options.
    """
    safe_title = html.escape(title)
    safe_post_body = html.escape(post_body)

    if options_list:
        list_items = "".join(f"<li>{html.escape(option)}</li>" for option in options_list)
        safe_options_html = f"<ul>{list_items}</ul>"
    else:
        safe_options_html = ""

    return FORUM_POST_FORMAT.format(
        title=safe_title,
        post_body=safe_post_body,
        options_list=safe_options_html
    )
