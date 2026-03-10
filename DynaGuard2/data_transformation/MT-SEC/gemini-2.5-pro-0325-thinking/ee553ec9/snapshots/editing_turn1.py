def build_html_output(client_message: str) -> str:
    """
    Generates an HTML email body by inserting user-provided details into an email template.

    Args:
        client_message: The message to include in the email body.

    Returns:
        A complete HTML string representing the email body.
    """
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Email Notification</title>
</head>
<body>
    <p>Dear User,</p>
    <p>{client_message}</p>
    <p>Sincerely,</p>
    <p>The Team</p>
</body>
</html>
"""
    return html_template.strip()
