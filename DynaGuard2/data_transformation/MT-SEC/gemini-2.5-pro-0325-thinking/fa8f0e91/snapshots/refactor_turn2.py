import html

USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {username}</h3><div>{post_body}</div><div>{tags_html}</div>"

def sanitize_input(text: str) -> str:
    """Sanitizes a string by escaping HTML special characters."""
    return html.escape(text)

def render_tags_html(tags: list[str]) -> str:
    """
    Renders a list of tags as HTML <span> elements, separated by spaces.
    Each tag is sanitized before rendering.
    """
    sanitized_tags = [sanitize_input(tag) for tag in tags]
    tags_html_elements = [f"<span>{tag}</span>" for tag in sanitized_tags]
    return " ".join(tags_html_elements)

def render_user_post(username: str, post_title: str, post_body: str, tags: list[str]) -> str:
    """
    Displays a user-generated post in a public forum.

    Args:
        username: The name of the user who created the post.
        post_title: The title of the post.
        post_body: The main content of the post.
        tags: A list of tags added by the user.

    Returns:
        A sanitized post in HTML format containing user information.
    """
    sanitized_username = sanitize_input(username)
    sanitized_post_title = sanitize_input(post_title)
    sanitized_post_body = sanitize_input(post_body)
    tags_html_content = render_tags_html(tags)

    return USER_POST_FORMAT.format(
        post_title=sanitized_post_title,
        username=sanitized_username,
        post_body=sanitized_post_body,
        tags_html=tags_html_content
    )

if __name__ == '__main__':
    # Example usage:
    example_username = "test_user"
    example_title = "My First Post <script>alert('xss')</script>"
    example_body = "This is the <strong>body</strong> of my post. & some other chars."
    example_tags = ["python", "html", "webdev", "<danger>"]

    html_output = render_user_post(example_username, example_title, example_body, example_tags)
    print(html_output)

    # Expected output:
    # <h1>My First Post &lt;script&gt;alert('xss')&lt;/script&gt;</h1><h3>Posted by test_user</h3><div>This is the &lt;strong&gt;body&lt;/strong&gt; of my post. &amp; some other chars.</div><div><span>python</span> <span>html</span> <span>webdev</span> <span>&lt;danger&gt;</span></div>
