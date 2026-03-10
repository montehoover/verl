def format_message(template, name):
    return template.replace("{name}", name)

def create_greeting(name):
    template = "Hello, {name}!"
    return format_message(template, name)
