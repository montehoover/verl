from collections import defaultdict

def select_columns(records: list[dict], fields: list[str]) -> list[dict]:
    """
    Extracts specific fields from a list of records.

    Args:
        records: A list of dictionaries, where each dictionary represents a record.
        fields: A list of strings, where each string is a field name to extract.

    Returns:
        A list of dictionaries, where each dictionary contains only the specified
        fields from the original records. If a field does not exist in a record,
        it is omitted from the resulting dictionary for that record.
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

    fields_to_select = ["name", "age", "city"]
    
    selected_data = select_columns(data, fields_to_select)
    print("Selected data:")
    for row in selected_data:
        print(row)

    fields_to_select_2 = ["id", "occupation"]
    selected_data_2 = select_columns(data, fields_to_select_2)
    print("\nSelected data 2:")
    for row in selected_data_2:
        print(row)
    
    # Example with an empty record list
    empty_data = []
    selected_empty_data = select_columns(empty_data, fields_to_select)
    print("\nSelected data from empty list:")
    for row in selected_empty_data:
        print(row) # Should print nothing

    # Example with an empty fields list
    data_for_empty_fields = [{"id": 1, "name": "Dana"}]
    selected_data_empty_fields = select_columns(data_for_empty_fields, [])
    print("\nSelected data with empty fields list:")
    for row in selected_data_empty_fields:
        print(row) # Should print [{}]
