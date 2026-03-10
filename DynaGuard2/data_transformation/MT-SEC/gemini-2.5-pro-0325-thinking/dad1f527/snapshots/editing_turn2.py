def read_text_file(message: str) -> None:
    """
    Logs a message to a file named log.txt.

    Appends the message to the log file with a newline.

    Args:
        message: The message string to log.
    """
    with open('log.txt', 'a') as log_file:
        log_file.write(message + '\n')
