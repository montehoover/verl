def format_message(template, name):
    return template.replace("{name}", name)

def create_greeting(name):
    template = "Hello, {name}!"
    return format_message(template, name)

def compose_hello_command(recipient):
    base_command = "echo Hello, {recipient}!"
    return base_command.format(recipient=recipient)
