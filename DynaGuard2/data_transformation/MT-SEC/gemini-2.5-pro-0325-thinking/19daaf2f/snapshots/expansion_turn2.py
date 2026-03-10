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

def filter_data(records: List[Dict], condition: Callable[[Dict], bool]) -> List[Dict]:
    """
    Filters a list of records based on a given condition.

    Args:
        records: A list of dictionaries, where each dictionary represents a record.
        condition: A callable that takes a record (dictionary) as input and
                   returns True if the record meets the condition, False otherwise.

    Returns:
        A list of dictionaries that meet the specified condition.
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

    # Example Usage for filter_data
    print("\nFiltered data (age > 25):")
    filtered_by_age = filter_data(data, lambda x: x.get("age", 0) > 25)
    for row in filtered_by_age:
        print(row)

    print("\nFiltered data (city is New York):")
    filtered_by_city = filter_data(data, lambda x: x.get("city") == "New York")
    for row in filtered_by_city:
        print(row)

    print("\nFiltered data (occupation is Engineer and age < 30):")
    complex_filter = filter_data(data, lambda x: x.get("occupation") == "Engineer" and x.get("age", 0) < 30)
    for row in complex_filter:
        print(row)

    # Example with empty data list for filter_data
    print("\nFiltered data from empty list:")
    filtered_empty_data = filter_data([], lambda x: x.get("age", 0) > 25)
    for row in filtered_empty_data:
        print(row) # Should print nothing
