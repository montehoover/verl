import re

def extract_binary_metadata(binary_data_stream: bytes) -> dict:
    """
    Extracts embedded metadata or header information from a binary data stream.

    The actual extraction logic depends heavily on the specific format of the
    binary data stream (e.g., presence of magic numbers, length fields,
    specific encoding for metadata). This function serves as a placeholder
    and will need to be implemented according to the data format.

    Args:
        binary_data_stream: The binary data (as bytes) to process.

    Returns:
        A dictionary containing the extracted metadata keys and values.
        Currently returns an empty dictionary as a placeholder.
    """
    metadata = {}

    # TODO: Implement metadata extraction logic here.
    # This will depend on the specific format of your binary data.
    # For example, you might need to:
    # 1. Check for a magic number or signature at the beginning of the stream.
    # 2. Read a header section which might have a fixed or variable length
    #    (e.g., length specified by a field in the header itself).
    # 3. Parse the header data. This data could be simple key-value pairs,
    #    a structured format like JSON or XML, or custom binary fields.
    #
    # Example of how one might start, using the 'struct' module for fixed-format binary data:
    #
    # import struct # Should be at the top of the file if used
    #
    # # Hypothetical example: metadata consists of a version (2 bytes) and type (1 byte)
    # # at the beginning of the stream.
    # if len(binary_data_stream) >= 3: # Minimum length for this hypothetical header
    #     try:
    #         # 'H' for unsigned short (2 bytes), 'B' for unsigned char (1 byte)
    #         # '>' for big-endian byte order.
    #         version, data_type = struct.unpack('>HB', binary_data_stream[0:3])
    #         metadata['version'] = version
    #         metadata['data_type'] = data_type
    #
    #         # If there's more complex metadata, continue parsing here.
    #         # For example, if the type indicates how to parse the rest:
    #         # if data_type == 1:
    #         #     # Parse type 1 specific metadata
    #         #     pass
    #
    #     except struct.error as e:
    #         # This can happen if binary_data_stream is shorter than expected by unpack,
    #         # or if data is malformed.
    #         print(f"Error unpacking binary data: {e}")
    #         # Optionally, handle this error more gracefully or log it.
    # else:
    #     # Stream is too short to contain the expected header.
    #     print("Binary data stream is too short for expected metadata header.")

    return metadata


def categorize_content_type(data: bytes) -> str:
    """
    Recognizes content type (JSON, INI) by sampling start markers.

    Args:
        data: The binary data (as bytes) to process.

    Returns:
        A string indicating the content type ("JSON", "INI").

    Raises:
        ValueError: For unrecognized, insecure, or empty formats.
    """
    sample_size = 256  # How many bytes to sample from the start
    sample_bytes = data[:sample_size]

    try:
        sample_str = sample_bytes.decode('utf-8', errors='replace')
    except UnicodeDecodeError:
        # This case should be rare with errors='replace', but good to have.
        raise ValueError("Unrecognized content type: Not valid UTF-8 text")

    stripped_sample = sample_str.strip()

    if not stripped_sample:
        raise ValueError("Unrecognized content type: Empty or whitespace-only content")

    # 1. Check for JSON object start
    if stripped_sample.startswith('{'):
        return "JSON"

    # 2. Check for INI section start (e.g., "[section_name]")
    # Pattern: '[' followed by a name starting with a letter or underscore,
    # then letters, numbers, _, ., -. This helps distinguish from JSON arrays like [1,2,3] or ["item"].
    # The closing ']' part of the pattern is optional to handle short samples.
    ini_section_pattern = re.compile(r"^\s*\[\s*([a-zA-Z_][a-zA-Z0-9_.\-]*)(?:\s*\])?")
    if ini_section_pattern.match(stripped_sample):
        return "INI"

    # 3. Check for JSON array start
    # If it starts with '[' and wasn't identified as an INI section above.
    # This covers `[1,2,3]`, `["item"]`, `[]`, `[{...}]` etc.
    if stripped_sample.startswith('['):
        return "JSON"

    # 4. Check for INI key-value pair at the start (e.g., "key = value")
    # This applies if the file doesn't start with a section or JSON structure.
    # Key name must start with a letter or underscore.
    ini_key_value_pattern = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_.\-]*)\s*=")
    if ini_key_value_pattern.match(stripped_sample):
        return "INI"

    # If none of the above markers are found
    raise ValueError(f"Unrecognized content type. Sample: '{stripped_sample[:50]}...'")
