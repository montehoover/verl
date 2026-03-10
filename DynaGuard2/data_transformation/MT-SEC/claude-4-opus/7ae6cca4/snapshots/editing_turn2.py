def decode_serialized_data(data_bytes, format_string):
    try:
        decoded_string = data_bytes.decode(format_string)
        print(decoded_string)
    except Exception as e:
        print(f"Error: {e}")
