USER_POST_FORMAT = "<h1>{heading}</h1><h3>Posted by {poster}</h3><div>{body}</div>"

def create_forum_post(poster: str, heading: str, body: str):
    """
    Generates a basic HTML view for a user's post and saves it to a file.

    Args:
        poster: The author's name.
        heading: The post's title.
        body: The main content of the post.
    """
    html_content = USER_POST_FORMAT.format(poster=poster, heading=heading, body=body)
    
    # Generate a simple filename from the heading
    # Replace spaces with underscores and remove non-alphanumeric characters for a safe filename
    filename_safe_heading = "".join(c if c.isalnum() or c in (' ', '_') else '' for c in heading).replace(' ', '_')
    if not filename_safe_heading: # Handle cases where heading might become empty
        filename_safe_heading = "post"
    filename = f"{filename_safe_heading}.html"
    
    with open(filename, "w") as f:
        f.write(html_content)
    # The request asked not to output natural language commentary,
    # but typically we would print the filename here or return it.
    # For now, just saving the file as requested.
