FORUM_POST_FORMAT = "<h2>{title}</h2><p>{post_body}</p>{options_list}"


def generate_post_html(title: str, content: str) -> str:
    """
    Generates an HTML representation of a forum post.

    Args:
        title: The title of the post.
        content: The content of the post.

    Returns:
        An HTML string representing the post.
    """
    return f"<div><h2>{title}</h2><p>{content}</p></div>"


def format_options_html(options: list[str]) -> str:
    """
    Formats a list of options into an HTML unordered list.

    Args:
        options: A list of strings, where each string is an option.

    Returns:
        An HTML string representing an unordered list of the options.
    """
    if not options:
        return "<ul></ul>"
    list_items = "".join([f"<li>{option}</li>" for option in options])
    return f"<ul>{list_items}</ul>"


def build_forum_post(title: str, post_body: str, options_list: list[str]) -> str:
    """
    Generates a complete HTML view of a forum post, including title, body, and a list of options.

    Args:
        title: The title of the forum post.
        post_body: The main content of the forum post.
        options_list: A list of strings, where each string is an option to be displayed.

    Returns:
        A string containing the complete HTML for the forum post.
    """
    formatted_options = format_options_html(options_list)
    return FORUM_POST_FORMAT.format(title=title, post_body=post_body, options_list=formatted_options)
