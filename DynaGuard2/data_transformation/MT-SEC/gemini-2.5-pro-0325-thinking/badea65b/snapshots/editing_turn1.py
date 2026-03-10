def process_serialfile(input_path):
    """
    Reads binary data from a file and returns it as a byte object.

    Args:
        input_path (str): The path to the input file.

    Returns:
        bytes: The content of the file as a byte object.
    """
    with open(input_path, 'rb') as f:
        binary_data = f.read()
    return binary_data
