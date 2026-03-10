def log_message(message: str) -> None:
    """
    Logs a message to a file named log.txt.

    Args:
        message: The message string to log.
    """
    with open('log.txt', 'a') as file:
        file.write(message + '\n')
