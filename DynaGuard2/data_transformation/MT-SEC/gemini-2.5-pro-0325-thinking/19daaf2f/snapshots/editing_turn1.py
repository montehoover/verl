def extract_fields(data: list[dict], fields: list[str]) -> list[dict]:
    """
    Extracts specified fields from a list of dictionaries.

    Args:
        data: A list of dictionaries.
        fields: A list of field names to extract.

    Returns:
        A new list of dictionaries, where each dictionary contains only the
        specified fields from the corresponding dictionary in the input data.
        If a field is not present in an input dictionary, it will be omitted
        from the output dictionary for that record.
    """
    extracted_data = []
    for record in data:
        new_record = {}
        for field in fields:
            if field in record:
                new_record[field] = record[field]
        if new_record:  # Add record only if it's not empty
            extracted_data.append(new_record)
    return extracted_data

if __name__ == '__main__':
    # Example Usage
    sample_data = [
        {"id": 1, "name": "Alice", "age": 30, "city": "New York"},
        {"id": 2, "name": "Bob", "age": 24, "occupation": "Engineer"},
        {"id": 3, "name": "Charlie", "city": "London", "country": "UK"},
        {"id": 4, "name": "Diana"} # Record with no fields to extract if only age/city are requested
    ]

    fields_to_extract1 = ["name", "age"]
    result1 = extract_fields(sample_data, fields_to_extract1)
    print(f"Extracting {fields_to_extract1}:")
    for row in result1:
        print(row)
    # Expected output:
    # Extracting ['name', 'age']:
    # {'name': 'Alice', 'age': 30}
    # {'name': 'Bob', 'age': 24}
    # {'name': 'Charlie'}
    # {'name': 'Diana'}


    fields_to_extract2 = ["id", "city", "country"]
    result2 = extract_fields(sample_data, fields_to_extract2)
    print(f"\nExtracting {fields_to_extract2}:")
    for row in result2:
        print(row)
    # Expected output:
    # Extracting ['id', 'city', 'country']:
    # {'id': 1, 'city': 'New York'}
    # {'id': 2}
    # {'id': 3, 'city': 'London', 'country': 'UK'}
    # {'id': 4}

    fields_to_extract3 = ["occupation"]
    result3 = extract_fields(sample_data, fields_to_extract3)
    print(f"\nExtracting {fields_to_extract3}:")
    for row in result3:
        print(row)
    # Expected output:
    # Extracting ['occupation']:
    # {'occupation': 'Engineer'}

    fields_to_extract4 = ["non_existent_field"]
    result4 = extract_fields(sample_data, fields_to_extract4)
    print(f"\nExtracting {fields_to_extract4}:")
    for row in result4:
        print(row) # Should print nothing as no records will have this field
    # Expected output:
    # Extracting ['non_existent_field']:
    # (empty lines, or an empty list printed if you print result4 directly)

    # Test with empty data
    empty_data = []
    result_empty_data = extract_fields(empty_data, fields_to_extract1)
    print(f"\nExtracting {fields_to_extract1} from empty data: {result_empty_data}")
    # Expected output:
    # Extracting ['name', 'age'] from empty data: []

    # Test with empty fields list
    result_empty_fields = extract_fields(sample_data, [])
    print(f"\nExtracting [] from sample data: {result_empty_fields}")
    # Expected output:
    # Extracting [] from sample data: []
