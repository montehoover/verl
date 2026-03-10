import json
from typing import Union, Any

def validate_byte_stream(stream: Union[bytes, bytearray]) -> bool:
    """
    Validates if the given byte stream is non-empty and not corrupted.

    Args:
        stream: The byte stream to validate. It should be a bytes or bytearray object.

    Returns:
        True if the stream is a non-empty bytes or bytearray object, False otherwise.
    """
    if not isinstance(stream, (bytes, bytearray)):
        # Not a valid byte stream type
        return False

    if not stream:
        # Empty stream
        return False

    # At this point, the stream is a non-empty bytes or bytearray object.
    # More sophisticated "corruption" checks (e.g., checksum, magic numbers)
    # would depend on the specific format of the byte stream and are not
    # implemented in this generic function.
    return True

def detect_format(stream: Union[bytes, bytearray]) -> str:
    """
    Detects the format of a byte stream (JSON, HTML, XML).

    Args:
        stream: The byte stream to analyze.

    Returns:
        A string indicating the format ("JSON", "HTML", "XML").

    Raises:
        ValueError: If the stream is empty, format is unrecognized, or potentially unsafe.
    """
    if not validate_byte_stream(stream):
        raise ValueError("Invalid or empty byte stream provided.")

    # Decode a small prefix of the stream for inspection
    # Attempt to decode as UTF-8, common for these formats.
    # Limit the prefix size to avoid reading too much data.
    prefix_size = 100  # Number of bytes to inspect
    try:
        # Ensure stream is bytes for slicing and decoding
        if isinstance(stream, bytearray):
            stream_bytes = bytes(stream)
        else:
            stream_bytes = stream
        
        prefix_str = stream_bytes[:prefix_size].decode('utf-8', errors='ignore').lstrip()
    except Exception as e:
        raise ValueError(f"Could not decode stream prefix for format detection: {e}")

    if not prefix_str:
        raise ValueError("Stream content is empty after decoding and stripping whitespace.")

    # Check for JSON
    if prefix_str.startswith('{') or prefix_str.startswith('['):
        return "JSON"

    # Check for XML (more specific than HTML)
    if prefix_str.lower().startswith('<?xml'):
        return "XML"
    
    # Check for HTML (common doctype or html tag)
    # HTML can also be XML (XHTML), so XML check should ideally be more robust
    # or come after a more specific HTML check if needed.
    # For simplicity, we check common HTML starts.
    lower_prefix = prefix_str.lower()
    if lower_prefix.startswith('<!doctype html>') or lower_prefix.startswith('<html>'):
        return "HTML"

    # Fallback check for general XML/HTML-like structures (starts with a tag)
    # This is a less specific check.
    if prefix_str.startswith('<'):
        # Could be XML or HTML not caught by specific checks above.
        # For this example, let's assume XML if it starts with a generic tag
        # and wasn't identified as HTML's specific doctype/html tag.
        # A more robust solution might involve trying to parse with XML/HTML parsers.
        # However, the request is to identify based on initial characters.
        # If it starts with '<' and not '<?xml', '<html>', or '<!doctype html>',
        # it's more likely to be a generic XML document or a fragment.
        # We will classify as XML for now.
        # A more sophisticated check might look for closing tags or specific namespaces.
        if '/>' in prefix_str or '</' in prefix_str: # Basic check for tag structure
            return "XML" # Or potentially "HTML" if it's a fragment

    raise ValueError("Unrecognized or potentially unsafe byte stream format.")

accepted_formats = ["json", "html"]

def decode_data_stream(data: bytes, data_format: str) -> Any:
    """
    Securely deserializes a byte stream into its corresponding Python object.

    Args:
        data: The byte stream to deserialize.
        data_format: The format of the data (e.g., "json", "html").

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If deserialization of an unsupported or unsafe format is attempted,
                    or if deserialization fails.
    """
    if not isinstance(data, bytes):
        raise ValueError("Input data must be a byte stream.")

    normalized_format = data_format.lower()

    if normalized_format not in accepted_formats:
        raise ValueError(f"Unsupported or unsafe data format: {data_format}. Accepted formats are: {accepted_formats}")

    try:
        if normalized_format == "json":
            # Decode bytes to string (assuming UTF-8, common for JSON)
            data_str = data.decode('utf-8')
            return json.loads(data_str)
        elif normalized_format == "html":
            # For HTML, "deserialization" typically means parsing into a document tree.
            # However, returning a simple string representation is safer and often what's needed
            # if the goal is to get the content without complex object representation.
            # If a full DOM object were required, a library like BeautifulSoup would be used,
            # but that's beyond simple "deserialization" into generic Python objects.
            # Returning the decoded string is a safe default.
            return data.decode('utf-8')
        # XML was in detect_format but not in accepted_formats for decode_data_stream
        # If XML were to be supported, a safe XML parser (e.g., defusedxml) would be essential.
        # else:
            # This case should not be reached if normalized_format is in accepted_formats
            # but as a safeguard:
            # raise ValueError(f"Internal error: Format {normalized_format} was accepted but not handled.")

    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to decode data for format {data_format}: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to deserialize JSON data: {e}")
    except Exception as e:
        # Catch any other unexpected errors during processing
        raise ValueError(f"An unexpected error occurred during deserialization for format {data_format}: {e}")
    
    # Fallback, though theoretically unreachable if logic is correct
    raise ValueError(f"Deserialization for format '{data_format}' not implemented or failed.")
