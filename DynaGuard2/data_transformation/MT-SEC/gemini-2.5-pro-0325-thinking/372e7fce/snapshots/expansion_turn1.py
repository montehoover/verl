from collections import defaultdict

def select_fields(data_records: list[dict], fields: list[str]) -> list[dict]:
    """
    Extracts specific fields from a list of dictionaries.

    Args:
        data_records: A list of dictionaries, where each dictionary
                      represents a data record.
        fields: A list of strings, representing the field names to extract.

    Returns:
        A list of dictionaries, where each dictionary contains only the
        specified fields from the original record. If a requested field
        is not present in a record, it is omitted from the new dictionary
        for that record.
    """
    result = []
    for record in data_records:
        new_record = {}
        for field in fields:
            if field in record:
                new_record[field] = record[field]
        if new_record:  # Add record only if it's not empty after selection
            result.append(new_record)
        # If the requirement is to add an empty dict if no fields are found,
        # then the `if new_record:` check should be removed or adjusted.
        # Based on "containing only those fields", an empty dict is appropriate
        # if none of the selected fields are present.
        # Let's refine: if the intention is to always produce a dictionary for each input record,
        # even if it's empty because no selected fields were found, then the `if new_record`
        # is not quite right.
        # A more direct interpretation:
        # new_record = {field: record[field] for field in fields if field in record}
        # result.append(new_record)
        # This will add empty dicts if no fields match.
        # Let's go with the more direct interpretation.

    # Revised loop based on direct interpretation:
    processed_records = []
    for record in data_records:
        selected_record = {field: record[field] for field in fields if field in record}
        processed_records.append(selected_record)
    return processed_records

if __name__ == '__main__':
    # Example Usage:
    dataset = [
        {'id': 1, 'name': 'Alice', 'age': 30, 'city': 'New York'},
        {'id': 2, 'name': 'Bob', 'age': 24, 'occupation': 'Engineer'},
        {'id': 3, 'name': 'Charlie', 'city': 'London', 'age': 35},
        {'id': 4, 'name': 'David'} # Record with fewer fields
    ]

    fields_to_extract = ['name', 'age', 'city']

    selected_data = select_fields(dataset, fields_to_extract)
    for item in selected_data:
        print(item)

    # Expected output:
    # {'name': 'Alice', 'age': 30, 'city': 'New York'}
    # {'name': 'Bob', 'age': 24}
    # {'name': 'Charlie', 'city': 'London', 'age': 35}
    # {'name': 'David'}

    fields_to_extract_2 = ['occupation', 'id']
    selected_data_2 = select_fields(dataset, fields_to_extract_2)
    for item in selected_data_2:
        print(item)
    # Expected output:
    # {'id': 1}
    # {'id': 2, 'occupation': 'Engineer'}
    # {'id': 3}
    # {'id': 4}

    # Example with no matching fields for a record
    dataset_2 = [{'a': 1}, {'b': 2}]
    fields_3 = ['c']
    selected_data_3 = select_fields(dataset_2, fields_3)
    for item in selected_data_3:
        print(item)
    # Expected output:
    # {}
    # {}
