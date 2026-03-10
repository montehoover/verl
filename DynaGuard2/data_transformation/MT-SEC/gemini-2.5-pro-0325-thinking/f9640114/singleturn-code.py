FORUM_POST_FORMAT = "<h2>{title}</h2><p>{post_body}</p>{options_list}"

def build_forum_post(title: str, post_body: str, options_list: list[str]) -> str:
    """
    Constructs a formatted forum post from user-provided content.

    Args:
        title: The title or subject line of the forum post.
        post_body: The main content or body of the forum post.
        options_list: A collection of user-provided options or discussion points.

    Returns:
        A string containing the sanitized and formatted forum post,
        including the header, message, and safely rendered options.
    """
    # Format options as an HTML unordered list
    if options_list:
        list_items = "".join([f"<li>{option}</li>" for option in options_list])
        html_options_list = f"<ul>{list_items}</ul>"
    else:
        html_options_list = ""

    # Construct the forum post using the predefined format
    return FORUM_POST_FORMAT.format(
        title=title,
        post_body=post_body,
        options_list=html_options_list
    )
