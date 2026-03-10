import string
import json
import csv
import xml.etree.ElementTree as ET

def is_ascii_printable(byte_stream: bytes) -> bool:
    """
    Verifies if a given byte stream consists solely of ASCII printable characters.

    Args:
        byte_stream: The byte stream to check.

    Returns:
        True if all bytes represent ASCII printable characters, False otherwise.
    """
    try:
        decoded_string = byte_stream.decode('ascii')
    except UnicodeDecodeError:
        # If it's not valid ASCII, it can't be ASCII printable
        return False

    for char_code in byte_stream:
        char = chr(char_code)
        if char not in string.printable:
            return False
    return True

def detect_format(byte_input: bytes) -> str:
    """
    Attempts to identify the serialization scheme of a byte input.

    Args:
        byte_input: The byte stream to analyze.

    Returns:
        The format's name (e.g., 'json', 'csv', 'xml').

    Raises:
        ValueError: If the format isn't identified, is empty, or deemed unsafe.
    """
    if not byte_input:
        raise ValueError("Input cannot be empty.")

    # Try decoding as UTF-8, a common encoding for text-based formats
    try:
        text_input = byte_input.decode('utf-8').strip()
    except UnicodeDecodeError:
        # If it's not valid UTF-8, it's unlikely to be JSON, CSV, or XML in their common forms
        raise ValueError("Cannot decode input as UTF-8, format unrecognized.")

    if not text_input: # handles cases like b"   "
        raise ValueError("Input contains only whitespace after decoding.")

    # Try JSON
    if text_input.startswith('{') or text_input.startswith('['):
        try:
            json.loads(text_input)
            return 'json'
        except json.JSONDecodeError:
            pass  # Not valid JSON, try next format

    # Try XML
    if text_input.startswith('<'):
        try:
            ET.fromstring(text_input)
            return 'xml'
        except ET.ParseError:
            pass  # Not valid XML, try next format

    # Try CSV
    # CSV detection can be tricky and might misinterpret other text data.
    # Sniffer needs a non-empty string.
    if text_input: # Ensure there's content to sniff
        try:
            # Sniffing requires string input, usually a sample.
            # We provide a few lines for better sniffing if available.
            sample = "\n".join(text_input.splitlines()[:5]) # Use up to 5 lines for sniffing
            csv.Sniffer().sniff(sample)
            # Further checks could be added here, e.g., trying to parse with csv.reader
            # For simplicity, if sniff is successful, we assume CSV.
            return 'csv'
        except csv.Error:
            pass  # Not valid CSV

    raise ValueError("Unknown or unsafe format")
