from typing import Optional

def load_local_file(file_name: str, allowed_files: list[str]) -> Optional[str]:
    """
    Load a file from the local system and return its content as a string,
    but only if the file name is in the provided list of allowed files.

    Args:
        file_name: The name or path of the file to load.
        allowed_files: A list of file names/paths that are allowed to be loaded.

    Returns:
        The contents of the file as a string if the file is allowed; otherwise, None.

    Raises:
        FileNotFoundError: If the file is allowed but does not exist.
    """
    if file_name not in allowed_files:
        return None

    with open(file_name, 'r', encoding='utf-8') as f:
        return f.read()
