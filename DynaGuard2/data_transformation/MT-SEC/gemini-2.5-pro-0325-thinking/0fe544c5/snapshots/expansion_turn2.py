from collections import defaultdict
from typing import List, Dict, Any, Callable

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


def apply_filter(data: List[Dict[str, Any]], condition: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
    """
    Filters a list of records based on a given condition.

    Args:
        data: A list of dictionaries, where each dictionary represents a record.
        condition: A callable that takes a record (dictionary) as input and
                   returns True if the record satisfies the condition, False otherwise.

    Returns:
        A list of dictionaries containing only the records that satisfy the condition.
    """
    filtered_data: List[Dict[str, Any]] = []
    for record in data:
        if condition(record):
            filtered_data.append(record)
    return filtered_data

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

    # Example Usage for apply_filter:
    # Filter condition: age > 25 and city is 'New York' or 'London'
    def filter_condition(record: Dict[str, Any]) -> bool:
        # Handle missing keys gracefully by providing default values for comparison
        age = record.get('age')
        city = record.get('city')

        # Condition 1: age is present and greater than 25
        cond1 = age is not None and age > 25
        # Condition 2: city is present and is 'New York' or 'London'
        cond2 = city is not None and city in ['New York', 'London']
        
        # Example of combining conditions (e.g., age > 25 AND (city is 'New York' OR city is 'London'))
        # For this specific example, let's filter for records where age > 25
        # AND (the city is 'New York' OR the occupation is 'Engineer')
        
        occupation = record.get('occupation')
        
        # More complex filter: (age > 25 AND city == 'New York') OR occupation == 'Engineer'
        filter_logic = (age is not None and age > 25 and city == 'New York') or \
                       (occupation == 'Engineer')
        return filter_logic

    filtered_dataset = apply_filter(dataset, filter_condition)
    print("\nFiltered Dataset (age > 25 AND city == 'New York' OR occupation == 'Engineer'):")
    for row in filtered_dataset:
        print(row)

    # Example: Filter for records where 'city' is 'London'
    london_records = apply_filter(dataset, lambda record: record.get('city') == 'London')
    print("\nFiltered Dataset (city is 'London'):")
    for row in london_records:
        print(row)

    # Example with a filter that matches no records
    no_match_filter = apply_filter(dataset, lambda record: record.get('name') == 'Unknown')
    print(f"\nResult for filter with no matches: {no_match_filter}")

    # Example filtering on the previously selected_dataset
    # Filter condition: occupation is 'Engineer'
    engineers_from_selected = apply_filter(selected_dataset, lambda record: record.get('occupation') == 'Engineer')
    print("\nEngineers from selected_dataset (occupation is 'Engineer'):")
    for row in engineers_from_selected:
        print(row)
