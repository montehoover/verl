"""
Utilities for safely formatting and displaying user-submitted blog posts.

This module exposes a single public function: display_blog_entry(...).
It escapes HTML and preserves line breaks for safe and readable display.
"""

import logging
from html import escape


# Ensure BLOG_FORMAT is available. If a template called BLOG_FORMAT has been
# defined elsewhere (e.g., by the embedding application), we will use it.
# Otherwise, fall back to a safe default that includes headline and text slots.
try:
    BLOG_FORMAT  # type: ignore[name-defined]
except NameError:
    BLOG_FORMAT = "<h2>{headline}</h2><div>{text}</div>"


def display_blog_entry(headline: str, text: str) -> str:
    """
    Return a sanitized, HTML-formatted representation of a blog post.

    The function escapes HTML special characters to prevent injection and
    converts newline characters in the post body to <br> tags so that the
    original line-break formatting is preserved in the browser.

    Logging:
        This function initializes a dedicated logger on first use and logs:
        - The raw input headline and text received.
        - The final formatted HTML output.

        The logger emits human-readable messages with timestamps.

    Parameters:
        headline (str): The headline or title of the blog post.
        text (str): The full text content of the blog post.

    Returns:
        str: An HTML string formatted using BLOG_FORMAT with sanitized content.

    Notes:
        - Input values are coerced to strings and None is treated as an empty
          string to be defensive against unexpected inputs.
        - The global BLOG_FORMAT must contain "{headline}" and "{text}"
          placeholders.
    """
    # Initialize a human-readable logger local to this function.
    logger = logging.getLogger("blog.display_blog_entry")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    logger.info("Received request to format blog entry.")

    # Defensive coercion to string so that callers can pass non-str types
    # or None without causing errors. None becomes an empty string.
    headline_str = "" if headline is None else str(headline)
    text_str = "" if text is None else str(text)

    logger.info("Input headline: %s", headline_str)
    logger.info("Input text: %s", text_str)

    # Escape HTML special characters to prevent injection and broken markup.
    # Use quote=True to also escape quotation marks.
    safe_headline = escape(headline_str, quote=True)
    safe_text = escape(text_str, quote=True)

    # Normalize newlines and preserve them in HTML by converting to <br>.
    # This keeps the user's line breaks visible in the rendered page.
    safe_text = safe_text.replace("\r\n", "\n").replace("\r", "\n")
    safe_text = safe_text.replace("\n", "<br>")

    # Populate the provided HTML template with the sanitized values.
    formatted = BLOG_FORMAT.format(headline=safe_headline, text=safe_text)

    logger.info("Formatted HTML output: %s", formatted)

    return formatted
