def build_html_output(client_message: str) -> str:
    """
    Generate an HTML email body by inserting user-provided details into an email template.
    
    Args:
        client_message: The message to personalize the email's content
        
    Returns:
        A complete HTML string with the user's message inside the email body
    """
    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Email Message</title>
</head>
<body>
    <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
            <td align="center">
                <table width="600" cellpadding="20" cellspacing="0">
                    <tr>
                        <td>
                            <h1>Important Message</h1>
                            <hr>
                            <div>
                                <p>{client_message}</p>
                            </div>
                            <hr>
                            <p>Thank you for your attention.</p>
                            <p>Best regards,<br>The Team</p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>"""
    
    return html_template
