def read_text_file(file_path: str) -> str:
    """
    Read the content of a text file and return it as a string.

    Args:
        file_path: Path to the text file.

    Returns:
        The file contents as a string.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()
