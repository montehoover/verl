from typing import List, Optional

def load_local_file(file_name: str, allowed_files: List[str]) -> Optional[str]:
    """
    Loads a file from the local system if it is in the allowed list.

    Args:
        file_name: The name of the file to be loaded.
        allowed_files: A list of file names that are allowed to be loaded.

    Returns:
        The content of the file as a string if the file is allowed and found,
        otherwise None if the file is not in the allowed list.

    Raises:
        FileNotFoundError: If the file is in the allowed list but does not exist.
    """
    if file_name not in allowed_files:
        return None
    try:
        with open(file_name, 'r') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File '{file_name}' not found.")
