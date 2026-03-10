def assemble_greet_command(name_of_user: str):
    # Validate input contains only safe characters
    if not all(c.isalnum() or c.isspace() for c in name_of_user):
        raise ValueError("name_of_user must only contain letters, numbers, and spaces")
    
    base_command = "echo Hello, {name_of_user}!"
    return base_command.format(name_of_user=name_of_user)
