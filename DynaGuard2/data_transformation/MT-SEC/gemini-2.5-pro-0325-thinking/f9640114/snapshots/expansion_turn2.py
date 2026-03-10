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
