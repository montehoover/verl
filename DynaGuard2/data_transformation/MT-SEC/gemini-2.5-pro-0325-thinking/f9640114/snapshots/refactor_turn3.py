import html


"""
Module to build formatted forum posts.

This module provides functions to construct HTML-formatted forum posts
from user-provided content, ensuring that all user input is sanitized
to prevent XSS vulnerabilities.
"""

# Module-level constant defining the HTML template for a forum post.
# Placeholders {title}, {post_body}, and {options_list} will be replaced.
FORUM_POST_FORMAT = "<h2>{title}</h2><p>{post_body}</p>{options_list}"


def build_forum_post(title: str, post_body: str, options_list: list[str]) -> str:
    """Constructs a sanitized and formatted forum post.

    This function takes a title, post body, and a list of options.
    It escapes the string inputs to prevent Cross-Site Scripting (XSS) attacks
    and then formats them into an HTML string using a predefined template.
    The options are rendered as an HTML unordered list.

    Args:
        title: The title or subject line for the forum post.
        post_body: The main content or body of the forum post.
        options_list: A list of strings, where each string represents an
                      option or discussion point. These will be list items.

    Returns:
        A string containing the HTML-formatted and sanitized forum post.
    """
    # Sanitize the title and post body to prevent XSS vulnerabilities.
    # html.escape converts characters like '<', '>', '&' to HTML entities.
    safe_title = html.escape(title)
    safe_post_body = html.escape(post_body)

    # Generate the HTML for the options list.
    # This helper function also handles sanitization of each option.
    safe_options_html = _create_options_html_list(options_list)

    # Populate the forum post template with the sanitized content.
    return FORUM_POST_FORMAT.format(
        title=safe_title,
        post_body=safe_post_body,
        options_list=safe_options_html
    )


def _create_options_html_list(options_list: list[str]) -> str:
    """Creates an HTML unordered list from a list of options.

    Each option provided in the list is HTML-escaped to ensure it is
    safe for rendering in an HTML context. If the `options_list` is empty
    or None, an empty string is returned, preventing an empty <ul> tag.

    Args:
        options_list: A list of strings, where each string is an option
                      to be included in the HTML list.

    Returns:
        An HTML string representing an unordered list (e.g., "<ul><li>Option 1</li></ul>").
        Returns an empty string if `options_list` is empty or None.
    """
    if not options_list:
        # If there are no options, return an empty string.
        # This avoids rendering an empty <ul></ul> structure in the final post.
        return ""

    # For each option, escape it and wrap it in <li> tags.
    # Then, join all list items into a single string.
    # Example: "<li>Escaped Option 1</li><li>Escaped Option 2</li>"
    list_items = "".join(f"<li>{html.escape(option)}</li>" for option in options_list)

    # Wrap the generated list items in <ul> tags to form a complete unordered list.
    return f"<ul>{list_items}</ul>"
