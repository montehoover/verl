from collections import defaultdict
from typing import List, Dict, Any

def select_columns(data: List[Dict[str, Any]], columns: List[str]) -> List[Dict[str, Any]]:
    """
    Extracts specified columns from a list of dictionaries.

    For each dictionary in the input list, a new dictionary is created containing
    only the keys specified in the 'columns' list. If a specified column
    is not present in an input dictionary, it will be included in the output
    dictionary with a value of None.

    Args:
        data: A list of dictionaries, where each dictionary represents a record.
        columns: A list of strings, where each string is a column name to select.

    Returns:
        A list of dictionaries, where each dictionary contains only the
        specified columns.
    """
    # Define the default value factory for missing columns.
    # Missing columns will be assigned a value of None.
    default_value_factory = lambda: None

    selected_data: List[Dict[str, Any]] = []
    for record in data:
        # Create a defaultdict from the current record.
        # This allows accessing keys that might be missing, returning the default value.
        record_with_defaults = defaultdict(default_value_factory)
        record_with_defaults.update(record)

        new_record: Dict[str, Any] = {}
        for column_name in columns:
            # Populate the new record with values for the specified columns.
            # If a column was not in the original record, defaultdict provides None.
            new_record[column_name] = record_with_defaults[column_name]
        selected_data.append(new_record)

    return selected_data

if __name__ == '__main__':
    # Example Usage:
    dataset = [
        {'id': 1, 'name': 'Alice', 'age': 30, 'city': 'New York'},
        {'id': 2, 'name': 'Bob', 'age': 24, 'occupation': 'Engineer'},
        {'id': 3, 'name': 'Charlie', 'city': 'London', 'age': 35}
    ]

    columns_to_select = ['id', 'name', 'city', 'occupation']

    selected_dataset = select_columns(dataset, columns_to_select)

    print("Original Dataset:")
    for row in dataset:
        print(row)

    print("\nSelected Columns Dataset:")
    for row in selected_dataset:
        print(row)

    # Example with empty data
    empty_data_result = select_columns([], columns_to_select)
    print(f"\nResult for empty data: {empty_data_result}")

    # Example with empty columns list
    empty_columns_result = select_columns(dataset, [])
    print(f"\nResult for empty columns list: {empty_columns_result}")
