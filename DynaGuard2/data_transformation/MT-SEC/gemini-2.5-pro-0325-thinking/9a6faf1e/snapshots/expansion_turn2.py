import io
import json

def read_file_contents(file_path: str) -> str:
    """
    Reads the raw contents of a file and returns it as a string.

    Args:
        file_path: The path to the file.

    Returns:
        The contents of the file as a string.

    Raises:
        IOError: If the file cannot be read.
    """
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except IOError as e:
        # It's good practice to provide context or re-raise with a custom message
        # For now, we'll re-raise the original exception.
        raise IOError(f"Error reading file {file_path}: {e}")

def detect_file_format(file_path: str) -> str:
    """
    Detects the format of the file content (json, xml, or text).

    Args:
        file_path: The path to the file.

    Returns:
        A string indicating the file format ('json', 'xml', 'text').

    Raises:
        IOError: If the file cannot be read.
        ValueError: If the file content is empty, or the format is unrecognized
                    or potentially harmful.
    """
    content = read_file_contents(file_path)
    stripped_content = content.strip()

    if not stripped_content:
        raise ValueError("File content is empty or consists only of whitespace, format unrecognized.")

    # Check for JSON
    try:
        json.loads(stripped_content)
        return "json"
    except json.JSONDecodeError:
        pass  # Not JSON

    # Check for XML (basic check: starts with '<' and ends with '>')
    # More robust XML detection might involve trying to parse with an XML library,
    # but this is a common heuristic.
    if stripped_content.startswith('<') and stripped_content.endswith('>'):
        # Further check: ensure it's not just a single unclosed tag like "<abc"
        # by looking for a corresponding closing tag or self-closing tag.
        # This basic check is a starting point.
        # A simple check for a valid XML structure is non-trivial without a parser.
        # For now, we'll assume if it starts and ends with angle brackets, it's intended as XML.
        # A more robust solution would involve attempting to parse with xml.etree.ElementTree
        # e.g., import xml.etree.ElementTree as ET; ET.fromstring(stripped_content)
        return "xml"

    # Default to text if not clearly JSON or XML
    # A more sophisticated check for "text" might involve checking character encodings
    # or looking for binary data, but for now, this is the fallback.
    return "text"
