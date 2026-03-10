import html

html_template = "<html><body><h1>Welcome!</h1><p>{user_content}</p></body></html>"

def render_html_page(visitor_message):
    """
    Generates an HTML response by embedding user content into a predefined HTML structure.

    Args:
        visitor_message: The string message from the user.

    Returns:
        The resulting HTML string with the user's content safely embedded.
    """
    # Escape the user message to prevent XSS vulnerabilities
    safe_visitor_message = html.escape(visitor_message)
    return html_template.format(user_content=safe_visitor_message)

if __name__ == '__main__':
    # Example usage:
    # To see the output, you could write this to a file and open it in a browser.
    # For example:
    # with open("rendered_page.html", "w") as f:
    #     f.write(render_html_page("Hello from the user!"))
    # print("HTML page 'rendered_page.html' created.")
    print(render_html_page("A message from our visitor!"))
    print(render_html_page("Another message with <script>alert('oops')</script> tags."))
