def read_text_file(message: str) -> None:
    """
    Append a message to the log file 'log.txt', followed by a newline.

    Args:
        message: The message to log.
    """
    with open("log.txt", "a", encoding="utf-8") as f:
        f.write(f"{message}\n")
