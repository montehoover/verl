import shlex

base_command = "echo Hello, {recipient}!"

def compose_hello_command(recipient):
    safe_recipient = shlex.quote(recipient)
    return base_command.format(recipient=safe_recipient)
