from collections import defaultdict
from typing import Callable, List, Dict, Any
import re
import operator
from functools import partial # Included as requested, though not directly used in this version


_OPERATORS = {
    '=': operator.eq,
    '!=': operator.ne,
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
}

def _parse_value(value_str: str) -> Any:
    """
    Parses a string value from a query into a Python type (str, int, float, bool).
    """
    value_str = value_str.strip()
    # Handle quoted strings
    if (value_str.startswith("'") and value_str.endswith("'")) or \
       (value_str.startswith('"') and value_str.endswith('"')):
        return value_str[1:-1]
    
    # Handle boolean literals
    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False
    
    # Handle numbers
    try:
        return int(value_str)
    except ValueError:
        try:
            return float(value_str)
        except ValueError:
            raise ValueError(f"Unrecognized or unquoted value: {value_str}. Use quotes for strings, or provide valid numbers/booleans.")

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

def run_sql_query(dataset: List[Dict], sql_query: str) -> List[Dict]:
    """
    Processes SQL-like commands (SELECT, WHERE, ORDER BY) on a list of dictionaries.

    Args:
        dataset: A list of dictionaries representing the data.
        sql_query: A string containing the SQL-like query.
                   Example: "SELECT name, age WHERE age > 25 ORDER BY name ASC"

    Returns:
        A list of dictionaries containing the query results.

    Raises:
        ValueError: If the query is malformed, uses unsupported features,
                    or an error occurs during processing.
    """
    query_lower = sql_query.lower()

    idx_select = query_lower.find("select ")
    idx_where = query_lower.find(" where ")
    idx_orderby = query_lower.find(" order by ")

    if idx_select == -1 or not sql_query.strip().lower().startswith("select "): # Ensure SELECT is at the beginning
        raise ValueError("Query must start with SELECT clause.")

    # Determine end of SELECT clause string
    end_select_str_idx = len(sql_query)
    if idx_where != -1:
        end_select_str_idx = min(end_select_str_idx, idx_where)
    if idx_orderby != -1:
        end_select_str_idx = min(end_select_str_idx, idx_orderby)
    
    select_fields_str = sql_query[idx_select + len("select "):end_select_str_idx].strip()
    if not select_fields_str:
        raise ValueError("SELECT clause cannot be empty.")

    # WHERE clause
    where_condition_str = None
    if idx_where != -1:
        if idx_orderby != -1 and idx_orderby < idx_where:
            raise ValueError("ORDER BY clause cannot precede WHERE clause.")
        
        start_where_str_idx = idx_where + len(" where ")
        end_where_str_idx = idx_orderby if idx_orderby != -1 else len(sql_query)
        
        where_condition_str = sql_query[start_where_str_idx:end_where_str_idx].strip()
        if not where_condition_str:
            raise ValueError("WHERE clause cannot be empty if specified.")

    # ORDER BY clause
    orderby_fields_str = None
    if idx_orderby != -1:
        if idx_where != -1 and idx_orderby < idx_where: # Should be caught above, defensive check
             raise ValueError("ORDER BY clause cannot precede WHERE clause.")
        
        orderby_fields_str = sql_query[idx_orderby + len(" order by "):].strip()
        if not orderby_fields_str:
            raise ValueError("ORDER BY clause cannot be empty if specified.")

    current_data = list(dataset) # Work on a copy

    # --- Process WHERE clause ---
    if where_condition_str:
        # Simple WHERE condition parser: "field operator value"
        # Example: "age > 25", "name = 'Alice'"
        # Does not support AND/OR or complex expressions.
        where_match = re.match(r"(\w+)\s*([<>=!]{1,2})\s*(.+)", where_condition_str, re.IGNORECASE)
        if not where_match:
            raise ValueError(f"Malformed WHERE clause: '{where_condition_str}'. Expected 'field operator value'.")

        field_name = where_match.group(1).strip()
        op_str = where_match.group(2).strip()
        value_str = where_match.group(3).strip()

        if op_str not in _OPERATORS:
            raise ValueError(f"Unsupported operator in WHERE clause: {op_str}")
        op_func = _OPERATORS[op_str]

        try:
            parsed_value = _parse_value(value_str)
        except ValueError as e:
            raise ValueError(f"Error parsing value in WHERE clause ('{value_str}'): {e}")

        def condition_func(record: Dict) -> bool:
            record_value = record.get(field_name)

            if record_value is None:
                return False # Comparisons with NULL (None) are generally false in SQL, except IS NULL.

            current_record_value = record_value
            # Type coercion logic for comparison
            if isinstance(parsed_value, (int, float)) and not isinstance(current_record_value, (int, float)):
                try:
                    current_record_value = float(current_record_value)
                except (ValueError, TypeError):
                    return False # Cannot convert record value to numeric for comparison
            elif isinstance(parsed_value, str) and not isinstance(current_record_value, str):
                return False # Strict type check: if query value is string, record value must be string
            elif isinstance(parsed_value, bool) and not isinstance(current_record_value, bool):
                 # Attempt to coerce record value to bool if query value is bool
                if isinstance(current_record_value, str):
                    if current_record_value.lower() == 'true': current_record_value = True
                    elif current_record_value.lower() == 'false': current_record_value = False
                    else: return False # String is not 'true'/'false'
                elif isinstance(current_record_value, (int, float)):
                    current_record_value = bool(current_record_value) # 0/0.0 is False, others True
                else:
                    return False # Cannot coerce to bool

            try:
                return op_func(current_record_value, parsed_value)
            except TypeError:
                 # Catch comparison errors between incompatible types (e.g. str > int)
                return False
        
        current_data = filter_data(current_data, condition_func)

    # --- Process ORDER BY clause ---
    if orderby_fields_str:
        sort_criteria = []
        parts = orderby_fields_str.split(',')
        for part in parts:
            part = part.strip()
            sort_field_match = re.match(r"(\w+)(?:\s+(ASC|DESC))?", part, re.IGNORECASE)
            if not sort_field_match:
                raise ValueError(f"Malformed ORDER BY part: '{part}'. Expected 'field [ASC|DESC]'.")
            
            field = sort_field_match.group(1)
            order_str = sort_field_match.group(2).upper() if sort_field_match.group(2) else "ASC"
            if order_str not in ["ASC", "DESC"]: # Should not happen with regex but defensive
                raise ValueError(f"Invalid ORDER BY direction: {order_str}")
            sort_criteria.append({'field': field, 'reverse': order_str == "DESC"})

        for criterion in reversed(sort_criteria): # Stable sort: apply sorts from least to most significant
            field_to_sort_by = criterion['field']
            reverse_order = criterion['reverse']

            def sort_key_func(record: Dict) -> Any:
                val = record.get(field_to_sort_by)
                if val is None:
                    # Define consistent sorting for None values (e.g., always first or last)
                    # Python's default sort places None before other types.
                    # To make it explicit or change behavior:
                    # return (float('-inf') if not reverse_order else float('inf')) # Makes None sort first in ASC, last in DESC
                    return (1, None) # Sort None after actual values, or (0, None) for before
                
                # Handle basic types for robust sorting
                if isinstance(val, (int, float, str, bool)):
                    return (0, val) # (type_priority, value)
                return (2, str(val)) # Fallback for other types, sort by string representation

            try:
                current_data.sort(key=sort_key_func, reverse=reverse_order)
            except TypeError as e:
                raise ValueError(f"Error sorting by field '{field_to_sort_by}': Incompatible data types in column for comparison. {e}")

    # --- Process SELECT clause ---
    if select_fields_str.strip() == "*":
        result_data = current_data
    else:
        fields_to_select = [f.strip() for f in select_fields_str.split(',')]
        if not all(f for f in fields_to_select if f): # Check for empty field names like "name,,age"
             raise ValueError("Malformed SELECT clause: contains empty or invalid field names.")
        result_data = select_columns(current_data, fields_to_select)

    return result_data

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

    # Example Usage for run_sql_query
    print("\n--- SQL Query Examples ---")
    dataset_for_sql = [
        {"id": 1, "name": "Alice", "age": 30, "city": "New York", "active": True},
        {"id": 2, "name": "Bob", "age": 24, "occupation": "Engineer", "city": "Paris", "active": False},
        {"id": 3, "name": "Charlie", "age": 35, "city": "London", "active": True},
        {"id": 4, "name": "David", "age": 24, "city": "New York", "occupation": "Artist", "active": True},
        {"id": 5, "name": "Eve", "age": 30, "city": "Paris", "occupation": "Doctor", "active": False},
    ]

    queries = [
        "SELECT name, age FROM dataset WHERE age > 25 ORDER BY age DESC, name ASC",
        "SELECT * WHERE city = 'New York'",
        "SELECT name, city, occupation WHERE active = true ORDER BY name",
        "SELECT id, name WHERE age <= 24",
        "SELECT name, city WHERE city != 'Paris' ORDER BY city ASC",
        "SELECT * ORDER BY age ASC, name DESC",
        "SELECT name", # Simple select
        "SELECT name, age WHERE occupation = 'Doctor'",
        "SELECT name WHERE name = 'NonExistent'", # Should be empty
    ]

    for sql_query in queries:
        print(f"\nExecuting query: {sql_query}")
        try:
            results = run_sql_query(dataset_for_sql, sql_query)
            if results:
                for row in results:
                    print(row)
            else:
                print("No results found.")
        except ValueError as e:
            print(f"Query Error: {e}")

    print("\n--- SQL Query Error Examples ---")
    error_queries = [
        "SELECT name WHER age > 25", # Malformed WHERE
        "SELECT name FROM dataset WHERE age > 25 ORDER BY non_existent_field", # Sort by non-existent field (handled by sort key returning None)
        "SELECT name, age WHERE age > 'twenty five'", # Invalid value type for age
        "SELECT name ORDER BY name DESC WHERE age > 20", # Clauses out of order
        "UPDATE name SET age = 30", # Unsupported operation
        "SELECT name WHERE city =", # Incomplete WHERE
        "SELECT name, FROM table", # Empty field in SELECT
    ]
    for sql_query in error_queries:
        print(f"\nExecuting query: {sql_query}")
        try:
            results = run_sql_query(dataset_for_sql, sql_query)
            for row in results:
                print(row)
        except ValueError as e:
            print(f"Query Error: {e}")
