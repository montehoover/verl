import json
import csv
import io

allowed_methods = ["json", "csv"]

def deserialize_data(data_stream: bytes, deserialization_method: str):
    """
    Deserializes a data stream into its corresponding Python object,
    ensuring only authorized deserialization techniques are used.

    Args:
        data_stream: bytes, encapsulates the serialized data.
        deserialization_method: str, the string identifying the method
                                 used for serialization (e.g., 'json', 'csv').

    Returns:
        The object that has been deserialized from the byte stream.

    Raises:
        ValueError: If the deserialization method is disallowed or not recognized.
    """
    if deserialization_method not in allowed_methods:
        raise ValueError(
            f"Deserialization method '{deserialization_method}' is not allowed. "
            f"Allowed methods are: {', '.join(allowed_methods)}"
        )

    if deserialization_method == "json":
        try:
            # Decode bytes to string before parsing JSON
            decoded_stream = data_stream.decode('utf-8')
            return json.loads(decoded_stream)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Error decoding data stream as UTF-8: {e}")
    
    elif deserialization_method == "csv":
        try:
            # Decode bytes to string and use StringIO for csv.reader
            decoded_stream = data_stream.decode('utf-8')
            # Use io.StringIO to treat the string as a file
            csv_file = io.StringIO(decoded_stream)
            reader = csv.reader(csv_file)
            # Convert reader to a list of lists
            return [row for row in reader]
        except csv.Error as e:
            raise ValueError(f"Invalid CSV data: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Error decoding data stream as UTF-8: {e}")
            
    else:
        # This case should ideally not be reached if allowed_methods check is comprehensive
        # and covers all implemented methods.
        raise ValueError(
            f"Deserialization method '{deserialization_method}' is recognized but not implemented."
        )

if __name__ == '__main__':
    # Example Usage:
    
    # JSON Example
    json_data = {"name": "Alice", "age": 30, "city": "New York"}
    serialized_json = json.dumps(json_data).encode('utf-8')
    
    try:
        deserialized_object_json = deserialize_data(serialized_json, "json")
        print("Deserialized JSON:", deserialized_object_json)
        assert deserialized_object_json == json_data
    except ValueError as e:
        print(f"Error deserializing JSON: {e}")

    # CSV Example
    csv_data_string = "name,age,city\nBob,25,London\nCharlie,35,Paris"
    serialized_csv = csv_data_string.encode('utf-8')
    
    try:
        deserialized_object_csv = deserialize_data(serialized_csv, "csv")
        print("Deserialized CSV:", deserialized_object_csv)
        expected_csv_output = [['name', 'age', 'city'], ['Bob', '25', 'London'], ['Charlie', '35', 'Paris']]
        assert deserialized_object_csv == expected_csv_output
    except ValueError as e:
        print(f"Error deserializing CSV: {e}")

    # Pickle Example (disallowed)
    try:
        # Attempting to use a disallowed method
        # For demonstration, let's imagine we have some pickle data (though we won't actually create it)
        dummy_pickle_data = b"some_pickle_data" 
        deserialize_data(dummy_pickle_data, "pickle")
    except ValueError as e:
        print(f"Error with disallowed method: {e}")
        assert "not allowed" in str(e)

    # Unknown method example
    try:
        deserialize_data(b"some_data", "xml")
    except ValueError as e:
        print(f"Error with unknown method: {e}")
        assert "not allowed" in str(e)

    # Invalid JSON data example
    invalid_json_data = b"{'name': 'Alice', 'age': 30,}" # Note the trailing comma (invalid in standard JSON)
    try:
        deserialize_data(invalid_json_data, "json")
    except ValueError as e:
        print(f"Error with invalid JSON: {e}")
        assert "Invalid JSON data" in str(e)

    # Invalid CSV data (if csv.Error can be triggered easily, e.g. by malformed quotes)
    # For simplicity, we'll assume valid CSV structure for now.
    # The current CSV implementation is quite robust for basic CSV.

    print("\nAll example tests completed.")
