import json # Ensure json is imported at the top level for process_serialfile

allowed_formats = ["json", "text"]

def read_file_content(file_path: str) -> bytes:
    """
    Reads the content of a file and returns it as bytes.

    Args:
        file_path: The path to the file.

    Returns:
        The content of the file as bytes.
    """
    with open(file_path, 'rb') as f:
        content = f.read()
    return content


def detect_format(data: bytes) -> str:
    """
    Detects the format of the given byte data.

    Args:
        data: The byte data to analyze.

    Returns:
        A string representing the detected format ('json', 'xml', 'plain text').

    Raises:
        ValueError: If the format is ambiguous or unrecognizable.
    """
    import json

    # Attempt to decode as UTF-8 for initial checks.
    # This is a common encoding, but specific checks might need other decodings.
    try:
        text_content = data.decode('utf-8').strip()
    except UnicodeDecodeError:
        # If it can't be decoded as UTF-8, it's unlikely to be JSON or XML in their typical forms.
        # We might consider it 'plain text' if it's binary but not specifically recognized,
        # or raise an error if binary isn't a trusted "plain text" type.
        # For now, let's assume non-UTF-8 decodable is not one of our target formats.
        raise ValueError("Cannot decode data as UTF-8, format unrecognized.")

    is_json = False
    is_xml = False

    # Check for JSON
    if text_content.startswith('{') and text_content.endswith('}'):
        try:
            json.loads(text_content)
            is_json = True
        except json.JSONDecodeError:
            pass  # Not valid JSON
    elif text_content.startswith('[') and text_content.endswith(']'):
        try:
            json.loads(text_content)
            is_json = True
        except json.JSONDecodeError:
            pass  # Not valid JSON

    # Check for XML (basic check)
    if text_content.startswith('<') and text_content.endswith('>'):
        # A more robust XML check would involve trying to parse it with an XML library,
        # but for a simple detection, checking start/end tags is a common first step.
        # We'll consider it XML if it starts with '<' and doesn't look like JSON.
        # This could be improved by looking for <?xml version="1.0"?> or a root tag.
        if not is_json: # Avoid classifying JSON-like strings that start with < (e.g. "<not json>") as XML
            # A simple heuristic: if it starts with < and ends with >, it's likely XML or HTML.
            # We'll call it 'xml' for now.
            is_xml = True


    if is_json and is_xml:
        # This case should be rare with the current checks but good to handle.
        raise ValueError("Ambiguous format: data appears to be both JSON and XML.")
    
    if is_json:
        return 'json'
    elif is_xml:
        # Further check to avoid misclassifying plain text that happens to start/end with < >
        if text_content.count('<') > 0 and text_content.count('>') > 0 and \
           text_content.find('>') > text_content.find('<'):
            return 'xml'
        else: # If it doesn't have balanced tags or looks like simple text with < >
            # Re-evaluate if it should be plain text
            pass


    # If not clearly JSON or XML, consider it plain text.
    # This is a fallback. More sophisticated checks could be added.
    # For now, if it's decodable UTF-8 and not JSON/XML, it's plain text.
    if not is_json and not is_xml:
        # Check if it's empty or whitespace, which can be plain text
        if not text_content:
            return 'plain text' # Empty string is plain text
        
        # A simple heuristic for plain text: does it contain typical text characters?
        # This is very basic. A more robust check might involve character distribution
        # or absence of binary-looking sequences if we hadn't decoded it yet.
        # Since we decoded to text_content, if it's not JSON/XML, it's plain text by default here.
        return 'plain text'

    raise ValueError("Unrecognizable or ambiguous format.")


def process_serialfile(input_path: str, format_type: str):
    """
    Securely deserializes data from a file based on a specified format.

    Args:
        input_path: The path to the serialized file.
        format_type: The format type of the serialized data (e.g., 'json', 'text').

    Returns:
        The deserialized Python object.

    Raises:
        ValueError: If the format_type is not in allowed_formats or if deserialization fails.
    """
    if format_type not in allowed_formats:
        raise ValueError(f"Untrusted or prohibited format: {format_type}. Allowed formats are: {allowed_formats}")

    file_content_bytes = read_file_content(input_path)

    if format_type == "json":
        try:
            # Assuming UTF-8 encoding for JSON files, which is standard.
            return json.loads(file_content_bytes.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON from {input_path}: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode file content as UTF-8 for JSON processing from {input_path}: {e}")
    elif format_type == "text":
        try:
            # For 'text', we return the decoded string content.
            return file_content_bytes.decode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode file content as UTF-8 for text processing from {input_path}: {e}")
    
    # This part should ideally not be reached if format_type is validated against allowed_formats
    # and all allowed_formats have a handler.
    raise ValueError(f"Unsupported format type '{format_type}' for deserialization.")
