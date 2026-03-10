"""
Module for composing HTML representations of forum posts.

This module provides functions to safely construct HTML from user-supplied
post data, including title, author, content, and tags. It emphasizes
XSS protection by sanitizing all user inputs.
"""
import html

# USER_POST_FORMAT defines the HTML structure for a forum post.
# Placeholders {headline}, {poster_name}, {text}, and {tags_html} will be
# filled with sanitized user data.
USER_POST_FORMAT = "<h1>{headline}</h1><h3>Posted by {poster_name}</h3><div>{text}</div><div>{tags_html}</div>"

def _sanitize_input(text_input: str) -> str:
    """
    Sanitizes a string by escaping special HTML characters.

    This function is a crucial part of preventing Cross-Site Scripting (XSS)
    vulnerabilities by ensuring that user-supplied text is safe to embed in HTML.

    Args:
        text_input: The string to be sanitized.

    Returns:
        The sanitized string with HTML special characters (e.g., <, >, &)
        replaced by their corresponding HTML entities (e.g., &lt;, &gt;, &amp;).
    """
    # Use html.escape to replace special characters like '<', '>', '&', and '"'.
    return html.escape(text_input)

def _create_tags_html(hashtags: list[str]) -> str:
    """
    Creates an HTML string where each tag is sanitized and enclosed in a <span> element.

    This function processes a list of raw tag strings, sanitizes each one to prevent
    XSS, and then wraps each sanitized tag in HTML <span> tags. The resulting
    HTML snippets are concatenated into a single string.

    Args:
        hashtags: A list of strings, where each string is a tag.

    Returns:
        A single HTML string composed of all tags, each sanitized and
        wrapped in <span></span>. For example, for input ['python', 'web'],
        it might return "<span>python</span><span>web</span>".
    """
    # Sanitize each tag in the list using a list comprehension and the _sanitize_input helper.
    sanitized_tags = [_sanitize_input(tag) for tag in hashtags]
    # Create a list of HTML strings, where each string is a sanitized tag wrapped in <span> tags.
    tags_html_list = [f"<span>{tag}</span>" for tag in sanitized_tags]
    # Join the list of HTML tag strings into a single string.
    return "".join(tags_html_list)

def compose_post_html(poster_name: str, headline: str, text: str, hashtags: list[str]) -> str:
    """
    Constructs an HTML representation of a user's forum post.

    This function takes the author's name, post headline, main content, and a list
    of tags as input. It sanitizes all user-provided text to prevent XSS attacks
    and then formats these components into a structured HTML string using a
    predefined template (USER_POST_FORMAT).

    Args:
        poster_name: The name of the post's author.
        headline: The headline of the user's post.
        text: The primary text content of the user's post.
        hashtags: A collection of tags related to the post.

    Returns:
        A string containing the XSS-protected HTML representation of the user's forum post.
    """
    # Sanitize the headline, poster's name, and main text to prevent XSS.
    sanitized_headline = _sanitize_input(headline)
    sanitized_poster_name = _sanitize_input(poster_name)
    sanitized_text = _sanitize_input(text)
    
    # Generate the HTML for the tags, ensuring each tag is also sanitized.
    tags_html_content = _create_tags_html(hashtags)

    # Populate the USER_POST_FORMAT template with the sanitized and formatted components.
    # Each argument is placed on a new line for improved readability, following PEP-8 guidelines.
    return USER_POST_FORMAT.format(
        headline=sanitized_headline,
        poster_name=sanitized_poster_name,
        text=sanitized_text,
        tags_html=tags_html_content
    )
