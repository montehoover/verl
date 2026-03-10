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
    result = []
    for record in data:
        new_record = {}
        for field in fields:
            if field in record:
                new_record[field] = record[field]
        if new_record:  # Add record only if it's not empty after filtering
            result.append(new_record)
    return result

if __name__ == '__main__':
    # Example Usage
    sample_data = [
        {"id": 1, "name": "Alice", "age": 30, "city": "New York"},
        {"id": 2, "name": "Bob", "age": 24, "occupation": "Engineer"},
        {"id": 3, "name": "Charlie", "city": "London", "country": "UK"},
        {"id": 4, "name": "Diana"} # Record with no fields to extract if fields are e.g. ["age", "city"]
    ]

    fields_to_extract = ["name", "city"]
    extracted_data = extract_fields(sample_data, fields_to_extract)
    print("Original Data:")
    for item in sample_data:
        print(item)
    print("\nExtracted Data (fields: {}):".format(fields_to_extract))
    for item in extracted_data:
        print(item)

    fields_to_extract_2 = ["age", "occupation"]
    extracted_data_2 = extract_fields(sample_data, fields_to_extract_2)
    print("\nExtracted Data (fields: {}):".format(fields_to_extract_2))
    for item in extracted_data_2:
        print(item)

    fields_to_extract_3 = ["country"]
    extracted_data_3 = extract_fields(sample_data, fields_to_extract_3)
    print("\nExtracted Data (fields: {}):".format(fields_to_extract_3))
    for item in extracted_data_3:
        print(item)

    # Example with no matching fields in any record
    fields_to_extract_4 = ["non_existent_field"]
    extracted_data_4 = extract_fields(sample_data, fields_to_extract_4)
    print("\nExtracted Data (fields: {}):".format(fields_to_extract_4))
    for item in extracted_data_4:
        print(item)
    if not extracted_data_4:
        print("No data extracted as fields were not found.")
