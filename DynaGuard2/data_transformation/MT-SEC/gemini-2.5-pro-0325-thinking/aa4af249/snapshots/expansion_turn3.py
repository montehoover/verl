from collections import defaultdict
from typing import Callable, List, Dict
import re
import operator
from functools import partial # Included as requested, though not directly used in this specific implementation path

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

# --- SQL Processing Functions ---

OPS = {
    '=': operator.eq,
    '!=': operator.ne,
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
}

def _parse_condition_value(value_str: str):
    """Converts a SQL-like literal string to a Python type."""
    value_str = value_str.strip()
    if (value_str.startswith("'") and value_str.endswith("'")) or \
       (value_str.startswith('"') and value_str.endswith('"')):
        return value_str[1:-1]  # String literal
    try:
        return int(value_str)  # Integer literal
    except ValueError:
        try:
            return float(value_str)  # Float literal
        except ValueError:
            raise ValueError(f"Condition value '{value_str}' must be a number or a quoted string.")

def _create_condition_callable(condition_str: str) -> Callable[[Dict], bool]:
    """Parses a simple WHERE condition string (e.g., 'age > 30') and returns a callable."""
    match = re.match(r"^\s*(\w+)\s*([!=<>]=?)\s*(.+)\s*$", condition_str.strip())
    if not match:
        raise ValueError(f"Malformed WHERE condition: '{condition_str}'")

    field, op_str, value_literal = match.groups()

    op_func = OPS.get(op_str)
    if not op_func:
        raise ValueError(f"Unsupported operator: '{op_str}' in condition '{condition_str}'")

    value = _parse_condition_value(value_literal)

    def condition_callable(record: Dict) -> bool:
        if field not in record:
            return False  # Field doesn't exist, condition cannot be met
        record_value = record[field]
        try:
            # Ensure compatible types for comparison if possible, or let comparison raise TypeError
            if isinstance(value, str) and not isinstance(record_value, str):
                # Attempt to compare string value with non-string record_value might be problematic
                # Depending on desired behavior, could convert record_value to str or raise error
                # For now, direct comparison is attempted.
                pass
            elif isinstance(value, (int, float)) and not isinstance(record_value, (int, float)):
                 # Comparing number with non-number.
                 pass # Let direct comparison proceed, may cause TypeError
            return op_func(record_value, value)
        except TypeError:
            # Incompatible types for the operation (e.g., comparing int with str for '<')
            return False
    return condition_callable

def process_sql_request(dataset_records: List[Dict], sql_statement: str) -> List[Dict]:
    """
    Processes a SQL-like query (SELECT, WHERE, ORDER BY) on a list of records.
    """
    original_sql = sql_statement
    sql_statement = sql_statement.strip()

    # Regex to parse "SELECT columns [WHERE condition] [ORDER BY field [ASC|DESC]]"
    # This enforces clause order: SELECT -> WHERE -> ORDER BY
    query_match = re.match(
        r"SELECT\s+(.+?)"                         # SELECT columns
        r"(?:\s+WHERE\s+(.+?))?"                  # Optional WHERE condition
        r"(?:\s+ORDER BY\s+([a-zA-Z0-9_]+)(?:\s+(ASC|DESC))?)?$",  # Optional ORDER BY field [ASC|DESC]
        sql_statement, re.IGNORECASE
    )

    if not query_match:
        raise ValueError(f"Malformed SQL query. Expected format: SELECT cols [WHERE cond] [ORDER BY field [ASC|DESC]]. Query: '{original_sql}'")

    columns_str, where_clause_str, order_by_field, order_by_direction = query_match.groups()

    columns_str = columns_str.strip()
    if where_clause_str:
        where_clause_str = where_clause_str.strip()
    # order_by_field and order_by_direction can be None

    processed_records = list(dataset_records)  # Work on a copy

    # 1. Apply WHERE clause
    if where_clause_str:
        try:
            condition_callable = _create_condition_callable(where_clause_str)
            processed_records = apply_filter(processed_records, condition_callable)
        except ValueError as e:
            raise ValueError(f"Error in WHERE clause '{where_clause_str}': {e}")

    # 2. Apply ORDER BY clause
    if order_by_field:
        order_by_field = order_by_field.strip()
        reverse_sort = (order_by_direction and order_by_direction.upper() == 'DESC')

        # Sort key that handles None values and attempts to sort mixed types robustly.
        # Nones are typically sorted first in ascending, last in descending.
        def sort_key(record):
            val = record.get(order_by_field)
            # (is_none, value) tuple ensures Nones group together.
            # str(val) for non-None allows comparison across some mixed types (e.g. int and str)
            # though this might not always be semantically ideal for all mixed-type scenarios.
            if val is None:
                return (True, None) # Nones first
            # Attempt to create a sortable representation
            # If types are truly incompatible (e.g. dict vs int), str() is a fallback.
            try:
                # This allows numbers to sort numerically, strings alphabetically
                # but doesn't make numbers and strings directly comparable to each other
                # without further logic if they are mixed under the same key.
                # Python 3 sort is stable with mixed types if they don't compare (e.g. int vs str).
                # Forcing str(val) makes everything comparable as strings.
                # A common SQL behavior is to treat types differently.
                # For simplicity, (val is None, val) is often used, relying on Python's type comparison rules.
                return (False, val)
            except TypeError: # Should not happen with (False, val) unless val itself is unorderable
                return (False, str(val))


        try:
            processed_records.sort(key=sort_key, reverse=reverse_sort)
        except TypeError as e:
            # This might occur if 'val' from sort_key contains uncomparable types (e.g. int and list)
            # that sort() cannot handle even with the tuple trick.
            raise ValueError(f"Error sorting by '{order_by_field}': Incompatible data types for comparison. {e}")


    # 3. Apply SELECT clause
    final_records: List[Dict]
    if columns_str == "*":
        if not processed_records:
            selected_fields_list = []
        else:
            # For SELECT *, gather all unique keys from the (filtered and sorted) records
            all_keys = set()
            for record in processed_records:
                all_keys.update(record.keys())
            selected_fields_list = sorted(list(all_keys)) # Consistent order
        final_records = select_columns(processed_records, selected_fields_list)
    else:
        raw_selected_fields = columns_str.split(',')
        selected_fields_list = []
        for field_part in raw_selected_fields:
            stripped_field = field_part.strip()
            if not stripped_field:
                raise ValueError(f"Malformed SELECT clause: contains empty field name in '{columns_str}'")
            selected_fields_list.append(stripped_field)
        final_records = select_columns(processed_records, selected_fields_list)

    return final_records


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

    # --- Example Usage for process_sql_request ---
    print("\n--- SQL Processing Examples ---")
    queries = [
        "SELECT name, age FROM data WHERE age > 25 ORDER BY name DESC",
        "SELECT id, city WHERE name = 'Alice'",
        "SELECT * WHERE city = 'London' ORDER BY age ASC",
        "SELECT occupation", # Selects 'occupation', other records will have it missing
        "SELECT name WHERE age <= 24",
        "SELECT * ORDER BY id DESC"
    ]

    for query in queries:
        print(f"\nExecuting query: {query}")
        try:
            results = process_sql_request(data, query)
            if results:
                for row in results:
                    print(row)
            else:
                print("No results found.")
        except ValueError as e:
            print(f"Query Error: {e}")

    print("\n--- SQL Error Handling Examples ---")
    error_queries = [
        "SELECT name FROM data WHERE age ! 30", # Malformed operator
        "SELECT name, FROM data", # Empty column name
        "SELECT name WHERE city = NonQuotedString", # Non-quoted string value
        "UPDATE name SET age = 30", # Unsupported statement
        "SELECT name ORDER BY age WHERE city = 'NY'", # Clauses out of order
    ]
    for query in error_queries:
        print(f"\nExecuting query: {query}")
        try:
            results = process_sql_request(data, query)
            for row in results:
                print(row)
        except ValueError as e:
            print(f"Query Error: {e}")
