import shlex

def construct_command(command_template):
    return command_template

def embed_user_name(command_template, user_name):
    sanitized_name = shlex.quote(user_name)
    return command_template.replace("{user_name}", sanitized_name)
