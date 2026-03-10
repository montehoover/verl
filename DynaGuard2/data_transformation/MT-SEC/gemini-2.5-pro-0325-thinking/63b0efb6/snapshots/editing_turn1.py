def extract_fields(data: list[dict], fields: list[str]) -> list[dict]:
    """
    Extracts specified fields from a list of dictionaries.

    Args:
        data: A list of dictionaries.
        fields: A list of field names to extract.

    Returns:
        A new list of dictionaries, where each dictionary contains only
        the specified fields from the original dictionaries.
    """
    result = []
    for record in data:
        new_record = {}
        for field in fields:
            if field in record:
                new_record[field] = record[field]
        if new_record:  # Add record only if it's not empty (e.g., if at least one field was found)
            result.append(new_record)
    return result

if __name__ == '__main__':
    # Example Usage
    sample_data = [
        {"id": 1, "name": "Alice", "age": 30, "city": "New York"},
        {"id": 2, "name": "Bob", "age": 24, "occupation": "Engineer"},
        {"id": 3, "name": "Charlie", "city": "London", "age": 35}
    ]

    fields_to_extract = ["name", "age"]
    extracted_data = extract_fields(sample_data, fields_to_extract)
    print("Extracted data (name, age):")
    for item in extracted_data:
        print(item)

    fields_to_extract_2 = ["id", "city", "occupation"]
    extracted_data_2 = extract_fields(sample_data, fields_to_extract_2)
    print("\nExtracted data (id, city, occupation):")
    for item in extracted_data_2:
        print(item)

    fields_to_extract_3 = ["non_existent_field"]
    extracted_data_3 = extract_fields(sample_data, fields_to_extract_3)
    print("\nExtracted data (non_existent_field):")
    for item in extracted_data_3:
        print(item) # Should be empty or records with only existing fields if any

    fields_to_extract_4 = []
    extracted_data_4 = extract_fields(sample_data, fields_to_extract_4)
    print("\nExtracted data (no fields):")
    for item in extracted_data_4:
        print(item) # Should be a list of empty dicts or an empty list depending on implementation

    empty_data = []
    extracted_data_empty = extract_fields(empty_data, fields_to_extract)
    print("\nExtracted data (empty input data):")
    for item in extracted_data_empty:
        print(item)
