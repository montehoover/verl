from html import escape

def build_html_output(client_message: str) -> str:
    """
    Generate a complete HTML email body, inserting the user-provided message.
    Uses only basic HTML tags without CSS or JavaScript.

    Args:
        client_message: The user's message to include in the email body.

    Returns:
        A complete HTML string containing the message in a designated section.
    """
    # Normalize and escape
    normalized = (client_message or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    safe_message = escape(normalized, quote=True).replace("\n", "<br>")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Your Message</title>
</head>
<body>
  <table width="100%" border="0" cellpadding="0" cellspacing="0">
    <tr>
      <td align="center">
        <table width="600" border="0" cellpadding="0" cellspacing="0">
          <tr>
            <td align="left">
              <h1>Thanks for contacting us</h1>
              <p>We received your message and will get back to you as soon as possible.</p>
              <hr>
              <h2>Your Message</h2>
              <div id="client-message">{safe_message}</div>
              <hr>
              <p>Best regards,<br>Customer Support</p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""
    return html
