import shlex

base_command = "echo Hello, {recipient}!"

def compose_hello_command(recipient: str) -> str:
    # Escape the recipient name to prevent shell injection
    safe_recipient = shlex.quote(recipient)
    # Format the command with the escaped recipient
    return base_command.format(recipient=safe_recipient)
