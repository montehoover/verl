def generate_command_with_header(value: str):
    """Generates a SQL command string with a user-defined header value."""
    return f'SELECT {value} FROM users;'
