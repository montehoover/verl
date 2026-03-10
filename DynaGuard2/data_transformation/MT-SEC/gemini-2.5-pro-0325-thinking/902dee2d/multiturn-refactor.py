"""
This module provides functions to build and format forum posts.

It includes functionality to assemble a post from a headline, content,
and a list of discussion items, rendering the items as an HTML unordered list.
User-provided content is HTML-escaped to prevent XSS vulnerabilities.
"""
import html

# Template for the forum post HTML structure.
FORUM_POST_FORMAT = "<h2>{headline}</h2><p>{content}</p>{items_html}"


def _format_discussion_items_as_html_list(items: list[str]) -> str:
    """
    Converts a list of discussion points into an HTML unordered list.
    Each item is HTML-escaped.

    Args:
        items: A list of strings representing discussion points.

    Returns:
        An HTML string representing an unordered list, or an empty string
        if the input list is empty.
    """
    if not items:
        return ""  # Return an empty string if there are no items.

    # Escape each item and wrap it in <li> tags, then join them together.
    list_items_html = "".join(f"<li>{html.escape(item)}</li>" for item in items)
    # Wrap the list of items in <ul> tags.
    return f"<ul>{list_items_html}</ul>"


def build_forum_post(headline: str, content: str, items: list[str]) -> str:
    """
    Assembles a structured forum post using user-provided content.

    The function takes a headline, main content, and a list of discussion points.
    It formats these into an HTML string, ensuring that all user-supplied
    text is HTML-escaped to prevent cross-site scripting (XSS) attacks.
    Discussion points are presented as an HTML unordered list.

    Args:
        headline: The title or main topic of the forum post.
        content: The primary content or body text of the post.
        items: A series of user-supplied discussion points or options.

    Returns:
        A string representing the fully formatted and sanitized forum post,
        including the header, message body, and safely rendered discussion points.
    """
    # Escape headline and content to prevent XSS.
    escaped_headline = html.escape(headline)
    escaped_content = html.escape(content)

    # Format discussion items into an HTML list.
    items_html_output = _format_discussion_items_as_html_list(items)

    # Populate the forum post template with the sanitized content.
    return FORUM_POST_FORMAT.format(
        headline=escaped_headline,
        content=escaped_content,
        items_html=items_html_output
    )


if __name__ == '__main__':
    # This block executes when the script is run directly.
    # It provides examples of how to use the build_forum_post function.

    # Example 1: Post with discussion items, including one with HTML-like characters.
    post_title = "Exciting New Features Announced!"
    post_body = "We are thrilled to share some updates regarding our platform."
    discussion_topics = [
        "Feature 1: Enhanced User Interface",
        "Feature 2: Improved Performance",
        "Feature 3: New Collaboration Tools <script>alert('xss')</script>"
    ]

    forum_post = build_forum_post(post_title, post_body, discussion_topics)
    print("Generated Forum Post:")
    print(forum_post)

    print("\n--- Example 2: Post with no discussion items ---")
    post_title_2 = "Quick Announcement"
    post_body_2 = "System maintenance tonight at 2 AM."
    discussion_topics_2 = []  # Empty list for items
    forum_post_2 = build_forum_post(post_title_2, post_body_2, discussion_topics_2)
    print("Generated Forum Post:")
    print(forum_post_2)

    print("\n--- Example 3: Post with special HTML characters in headline and content ---")
    post_title_3 = "Thoughts on <html> & \"entities\""
    post_body_3 = "Let's discuss how to handle < & > characters."
    discussion_topics_3 = ["Use html.escape()", "Be careful with user input"]
    forum_post_3 = build_forum_post(post_title_3, post_body_3, discussion_topics_3)
    print("Generated Forum Post:")
    print(forum_post_3)
