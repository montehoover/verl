def extract_fields(dataset, fields):
    """
    Extracts specified fields from a list of dictionaries.

    Args:
        dataset: A list of dictionaries.
        fields: A list of field names to extract.

    Returns:
        A new list of dictionaries, where each dictionary contains only the
        specified fields.
    """
    new_dataset = []
    for item in dataset:
        new_item = {}
        for field in fields:
            if field in item:
                new_item[field] = item[field]
        new_dataset.append(new_item)
    return new_dataset

if __name__ == '__main__':
    # Example Usage
    data = [
        {'name': 'Alice', 'age': 30, 'city': 'New York'},
        {'name': 'Bob', 'age': 24, 'city': 'San Francisco'},
        {'name': 'Charlie', 'age': 35, 'city': 'London', 'occupation': 'Engineer'}
    ]

    fields_to_extract = ['name', 'city']
    extracted_data = extract_fields(data, fields_to_extract)
    print("Original dataset:")
    for row in data:
        print(row)
    print("\nExtracted dataset (name and city):")
    for row in extracted_data:
        print(row)

    fields_to_extract_with_occupation = ['name', 'occupation']
    extracted_data_occupation = extract_fields(data, fields_to_extract_with_occupation)
    print("\nExtracted dataset (name and occupation):")
    for row in extracted_data_occupation:
        print(row)
