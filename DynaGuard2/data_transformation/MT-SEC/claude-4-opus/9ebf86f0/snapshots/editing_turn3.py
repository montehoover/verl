import subprocess

base_command = "echo Hello, {user_name}!"

def create_greeting(user_name):
    return f"Hello, {user_name}!"

def generate_greeting_shell_command(user_name):
    return f'echo "Hello, {user_name}!"'

def generate_greeting_command(user_name):
    return base_command.format(user_name=user_name)
