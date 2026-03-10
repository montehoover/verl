from collections import defaultdict
from typing import Callable, List, Dict

def select_columns(records: List[Dict], fields: List[str]) -> List[Dict]:
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

def apply_filter(records: List[Dict], condition: Callable[[Dict], bool]) -> List[Dict]:
    """
    Filters a list of records based on a given condition.

    Args:
        records: A list of dictionaries, where each dictionary represents a record.
        condition: A callable that takes a record (dictionary) as input and
                   returns True if the record meets the condition, False otherwise.

    Returns:
        A list of dictionaries containing only the records that meet the condition.
    """
    filtered_records = []
    for record in records:
        if condition(record):
            filtered_records.append(record)
    return filtered_records

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

    # Example Usage for apply_filter
    print("\nFiltering for age > 30:")
    filtered_by_age = apply_filter(data, lambda x: x.get("age", 0) > 30)
    for item in filtered_by_age:
        print(item)
    # Expected Output:
    # {'id': 3, 'name': 'Charlie', 'city': 'London', 'age': 35}

    print("\nFiltering for city == 'New York':")
    filtered_by_city = apply_filter(data, lambda x: x.get("city") == "New York")
    for item in filtered_by_city:
        print(item)
    # Expected Output:
    # {"id": 1, "name": "Alice", "age": 30, "city": "New York"}
