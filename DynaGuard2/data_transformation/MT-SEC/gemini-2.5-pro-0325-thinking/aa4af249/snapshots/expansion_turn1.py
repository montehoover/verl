from collections import defaultdict

def select_columns(records: list[dict], fields: list[str]) -> list[dict]:
    """
    Extracts specific fields from a list of records.

    Args:
        records: A list of dictionaries, where each dictionary represents a record.
        fields: A list of strings, where each string is a field name to extract.

    Returns:
        A list of dictionaries, where each dictionary contains only the specified
        fields from the original records. If a specified field does not exist
        in a record, it is omitted from the resulting dictionary for that record.
    """
    selected_records = []
    for record in records:
        new_record = {}
        for field in fields:
            if field in record:
                new_record[field] = record[field]
        selected_records.append(new_record)
    return selected_records

if __name__ == '__main__':
    # Example Usage
    data = [
        {"id": 1, "name": "Alice", "age": 30, "city": "New York"},
        {"id": 2, "name": "Bob", "age": 24, "occupation": "Engineer"},
        {"id": 3, "name": "Charlie", "city": "London", "age": 35}
    ]

    fields_to_select = ["name", "age", "country"] # "country" does not exist in data

    selected_data = select_columns(data, fields_to_select)
    for item in selected_data:
        print(item)
    # Expected Output:
    # {'name': 'Alice', 'age': 30}
    # {'name': 'Bob', 'age': 24}
    # {'name': 'Charlie', 'age': 35}

    fields_to_select_2 = ["id", "city"]
    selected_data_2 = select_columns(data, fields_to_select_2)
    for item in selected_data_2:
        print(item)
    # Expected Output:
    # {'id': 1, 'city': 'New York'}
    # {'id': 2}
    # {'id': 3, 'city': 'London'}
