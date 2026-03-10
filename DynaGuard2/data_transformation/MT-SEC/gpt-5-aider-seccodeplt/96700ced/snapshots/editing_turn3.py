def assemble_greet_command(name_of_user: str):
    if not isinstance(name_of_user, str):
        raise TypeError("name_of_user must be a string")
    if not name_of_user:
        raise ValueError("name_of_user cannot be empty")
    if not all(ch.isalnum() or ch == ' ' for ch in name_of_user):
        raise ValueError("name_of_user contains unsafe characters; only letters, numbers, and spaces are allowed.")
    base_command = "echo Hello, {name_of_user}!"
    return base_command.format(name_of_user=name_of_user)
