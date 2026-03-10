import html
import logging

# --- Configuration ---

# Configure basic logging
# This will output logs to the console at the INFO level and above.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# HTML template for formatting user posts.
# Placeholders: {post_title}, {username}, {post_body}, {tags_html}
USER_POST_FORMAT = "<h1>{post_title}</h1><h3>Posted by {username}</h3><div>{post_body}</div><div>{tags_html}</div>"


# --- Helper Functions ---

def sanitize_input(text: str) -> str:
    """
    Sanitizes a string by escaping HTML special characters to prevent XSS.

    Args:
        text: The input string to sanitize.

    Returns:
        The sanitized string with HTML special characters escaped.
    """
    logging.debug(f"Sanitizing input: '{text[:50]}...'") # Log a snippet of the text
    sanitized_text = html.escape(text)
    logging.debug(f"Sanitized output: '{sanitized_text[:50]}...'")
    return sanitized_text


def render_tags_html(tags: list[str]) -> str:
    """
    Renders a list of tags as HTML <span> elements, separated by spaces.
    Each tag is sanitized before rendering to ensure safety.

    Args:
        tags: A list of strings, where each string is a tag.

    Returns:
        An HTML string representing the tags, e.g., "<span>tag1</span> <span>tag2</span>".
        Returns an empty string if the input list is empty.
    """
    if not tags:
        logging.info("No tags provided, returning empty string for tags HTML.")
        return ""

    logging.info(f"Rendering HTML for tags: {tags}")
    # Sanitize each tag individually
    sanitized_tags = [sanitize_input(tag) for tag in tags]
    # Wrap each sanitized tag in a <span> HTML element
    tags_html_elements = [f"<span>{tag}</span>" for tag in sanitized_tags]
    # Join the <span> elements with a space
    rendered_html = " ".join(tags_html_elements)
    logging.debug(f"Rendered tags HTML: {rendered_html}")
    return rendered_html


# --- Main Function ---

def render_user_post(username: str, post_title: str, post_body: str, tags: list[str]) -> str:
    """
    Orchestrates the rendering of a user-generated post into a sanitized HTML string.

    This function takes user-provided details for a post, sanitizes each piece of
    information to prevent cross-site scripting (XSS) attacks, formats the tags
    into HTML, and then constructs the final HTML output using a predefined template.

    Args:
        username: The name of the user who created the post.
        post_title: The title of the post.
        post_body: The main content of the post.
        tags: A list of tags associated with the post.

    Returns:
        A string containing the sanitized HTML representation of the user post.
    """
    logging.info(f"Rendering post for user: '{username}', title: '{post_title[:50]}...'")

    # Sanitize all user-provided string inputs to prevent XSS
    sanitized_username = sanitize_input(username)
    sanitized_post_title = sanitize_input(post_title)
    sanitized_post_body = sanitize_input(post_body)

    # Render the tags list into an HTML string
    # This step also includes sanitization of each tag.
    tags_html_content = render_tags_html(tags)

    # Populate the HTML template with sanitized content
    post_html = USER_POST_FORMAT.format(
        post_title=sanitized_post_title,
        username=sanitized_username,
        post_body=sanitized_post_body,
        tags_html=tags_html_content
    )
    logging.info(f"Successfully rendered post for user: '{username}'")
    return post_html


# --- Example Usage ---

if __name__ == '__main__':
    logging.info("Starting example usage of render_user_post.")

    # Example data for a user post
    example_username = "test_user"
    example_title = "My First Post <script>alert('xss')</script>" # Potentially malicious title
    example_body = "This is the <strong>body</strong> of my post. & some other chars." # Body with HTML and special chars
    example_tags = ["python", "html", "webdev", "<danger>"] # Tags, one potentially malicious

    # Log the input data
    logging.info(f"Example username: {example_username}")
    logging.info(f"Example title: {example_title}")
    logging.info(f"Example body: {example_body}")
    logging.info(f"Example tags: {example_tags}")

    # Generate the HTML output for the post
    html_output = render_user_post(
        username=example_username,
        post_title=example_title,
        post_body=example_body,
        tags=example_tags
    )

    # Print the generated HTML to the console
    print("\n--- Generated HTML Output ---")
    print(html_output)
    print("--- End of HTML Output ---\n")

    logging.info("Example usage finished.")

    # Expected output (printed to console, not as a log message):
    # <h1>My First Post &lt;script&gt;alert('xss')&lt;/script&gt;</h1><h3>Posted by test_user</h3><div>This is the &lt;strong&gt;body&lt;/strong&gt; of my post. &amp; some other chars.</div><div><span>python</span> <span>html</span> <span>webdev</span> <span>&lt;danger&gt;</span></div>
