from collections import defaultdict
from typing import List, Dict, Any, Callable, Tuple
import re
import operator
from functools import partial # Included as requested, though not strictly used in this impl.

# Global operator map for WHERE clause parsing
OPERATOR_MAP = {
    '=': operator.eq, '==': operator.eq,
    '!=': operator.ne, '<>': operator.ne,
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le
}

# Regex for parsing a single condition in WHERE clause: column_name OPERATOR value
# Value can be: boolean, number (int/float), or quoted string
CONDITION_PATTERN = re.compile(
    r"(\w+)\s*"  # Column name
    r"([<>=!]+)\s*"  # Operator
    r"(true|false|\d+\.\d+|\d+|'.*?'|\".*?\")",  # Value (boolean, float, int, quoted string)
    re.IGNORECASE  # For TRUE/FALSE literals
)

# Main regex to parse the overall SQL query structure
SQL_QUERY_PATTERN = re.compile(
    r"SELECT\s+(?P<columns>.+?)\s*"
    r"(?:FROM\s+\w+\s*)?"  # Optional FROM clause (table name is ignored)
    r"(?:WHERE\s+(?P<where_clause>.+?))?\s*"
    r"(?:ORDER\s+BY\s+(?P<orderby_clause>.+?))?$",
    re.IGNORECASE | re.DOTALL # Ignore case for keywords, DOTALL for multi-line clauses
)


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

# --- SQL Query Handling Functions ---

def _parse_sql_value(val_str: str) -> Any:
    """Converts a SQL value string to its Python type (str, int, float, bool)."""
    val_lower = val_str.lower()
    if val_lower == 'true':
        return True
    if val_lower == 'false':
        return False
    
    if (val_str.startswith("'") and val_str.endswith("'")) or \
       (val_str.startswith('"') and val_str.endswith('"')):
        return val_str[1:-1]  # String literal
    
    try:
        if '.' in val_str:
            return float(val_str)  # Float literal
        else:
            return int(val_str)  # Integer literal
    except ValueError:
        # This should ideally not be reached if CONDITION_PATTERN is robust
        raise ValueError(f"Cannot parse value: {val_str}. Must be a number, boolean, or quoted string.")

def _create_condition_checker(condition_str: str) -> Callable[[Dict[str, Any]], bool]:
    """Parses a single condition string (e.g., 'age > 25') and returns a callable checker."""
    match = CONDITION_PATTERN.match(condition_str.strip())
    if not match:
        raise ValueError(f"Malformed condition segment: {condition_str}")

    col_name, op_str, val_str = match.groups()
    
    if op_str not in OPERATOR_MAP:
        raise ValueError(f"Unsupported operator: {op_str}")
    op_func = OPERATOR_MAP[op_str]
    
    parsed_query_value = _parse_sql_value(val_str)

    def check_record(record: Dict[str, Any]) -> bool:
        record_val = record.get(col_name)

        if record_val is None:
            # Most SQL comparisons with NULL are UNKNOWN/FALSE.
            # IS NULL / IS NOT NULL would require specific handling (not supported here).
            return op_str == '!=' if parsed_query_value is None else (op_str == '=' if parsed_query_value is None else False)


        try:
            if isinstance(parsed_query_value, bool):
                if isinstance(record_val, bool):
                    return op_func(record_val, parsed_query_value)
                # Allow int 0/1 to compare with bool
                elif isinstance(record_val, int) and record_val in (0, 1):
                    return op_func(bool(record_val), parsed_query_value)
                return False # Type mismatch for boolean comparison
            elif isinstance(parsed_query_value, str):
                return op_func(str(record_val), parsed_query_value)
            elif isinstance(parsed_query_value, (int, float)):
                # Convert record_val to float for numeric comparison to handle int/float mixes
                numeric_record_val = float(record_val)
                # Ensure query value is also float for consistent comparison
                numeric_query_value = float(parsed_query_value) 
                return op_func(numeric_record_val, numeric_query_value)
        except (ValueError, TypeError):
            return False # Failed to convert or compare
        return False # Should not be reached
    return check_record

def _parse_where_clause(where_str: str) -> Callable[[Dict[str, Any]], bool]:
    """Parses the WHERE clause string, supporting AND/OR logic (AND has higher precedence)."""
    if not where_str or not where_str.strip():
        return lambda record: True # No filter

    or_parts = [part.strip() for part in re.split(r'\s+OR\s+', where_str, flags=re.IGNORECASE)]
    
    compiled_or_groups = []
    for or_part_str in or_parts:
        if not or_part_str: continue # Skip empty parts from multiple ORs or trailing OR

        and_parts = [part.strip() for part in re.split(r'\s+AND\s+', or_part_str, flags=re.IGNORECASE)]
        
        current_and_checkers = []
        for and_part_str in and_parts:
            if not and_part_str: continue # Skip empty parts
            current_and_checkers.append(_create_condition_checker(and_part_str))
        
        if not current_and_checkers: continue

        # Closure to evaluate all AND conditions in this group
        def make_and_evaluator(checkers: List[Callable[[Dict[str, Any]], bool]]):
            def and_evaluator(record: Dict[str, Any]) -> bool:
                for checker in checkers:
                    if not checker(record):
                        return False
                return True
            return and_evaluator
        compiled_or_groups.append(make_and_evaluator(current_and_checkers))

    if not compiled_or_groups: # e.g. WHERE clause was just " " or "OR OR"
         return lambda record: True


    def combined_evaluator(record: Dict[str, Any]) -> bool:
        for or_group_evaluator in compiled_or_groups:
            if or_group_evaluator(record):
                return True # One OR group is true
        return False # All OR groups are false
    return combined_evaluator

def _get_sortable_value(value: Any):
    """Prepares a value for sorting, handling None and basic types consistently."""
    if value is None:
        return (0, None)  # None sorts first
    if isinstance(value, bool):
        return (1, int(value)) # Bools (as 0/1)
    if isinstance(value, (int, float)):
        return (2, value)  # Numbers
    if isinstance(value, str):
        return (3, value.lower())  # Strings (case-insensitive sort)
    return (4, str(value))  # Other types as strings

def _parse_orderby_clause(orderby_str: str) -> List[Tuple[str, bool]]:
    """Parses ORDER BY clause string (e.g., 'age DESC, name ASC') into sort criteria."""
    if not orderby_str or not orderby_str.strip():
        return []
    
    criteria = []
    parts = [part.strip() for part in orderby_str.split(',')]
    for part in parts:
        if not part: continue
        match = re.match(r"(\w+)\s*(ASC|DESC)?", part, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Malformed ORDER BY part: {part}")
        
        col_name = match.group(1)
        direction_str = match.group(2)
        
        is_descending = bool(direction_str and direction_str.upper() == 'DESC')
        criteria.append((col_name, is_descending))
    return criteria

def handle_sql_query(records: List[Dict[str, Any]], sql_command: str) -> List[Dict[str, Any]]:
    """
    Processes a SQL-like query with SELECT, WHERE, and ORDER BY clauses.

    Args:
        records: A list of dictionaries (the dataset).
        sql_command: The SQL query string.

    Returns:
        A list of dictionaries representing the query result.

    Raises:
        ValueError: If the SQL query is malformed or an operation fails.
    """
    match = SQL_QUERY_PATTERN.match(sql_command.strip())
    if not match:
        raise ValueError("Invalid SQL query format. Must be SELECT ... [WHERE ...] [ORDER BY ...]")

    query_parts = match.groupdict()
    select_columns_str = query_parts.get("columns")
    where_clause_str = query_parts.get("where_clause")
    orderby_clause_str = query_parts.get("orderby_clause")

    if not select_columns_str or not select_columns_str.strip():
        raise ValueError("SELECT clause cannot be empty.")

    current_records = list(records) # Work on a copy

    # 1. Apply WHERE clause
    if where_clause_str:
        filter_condition_func = _parse_where_clause(where_clause_str)
        current_records = apply_filter(current_records, filter_condition_func)

    # 2. Apply ORDER BY clause
    if orderby_clause_str:
        sort_criteria = _parse_orderby_clause(orderby_clause_str)
        if sort_criteria:
            # Python's sort is stable. Sort by each criterion in reverse order of application.
            for col_name, is_descending in reversed(sort_criteria):
                # Check if column exists in first record to avoid errors if column is totally missing
                # This is a simple check; a more robust way might be to check all records or schema
                if current_records and col_name not in current_records[0] and \
                   not any(col_name in r for r in current_records): # check if col_name exists in any record
                    # Silently ignore sort by non-existent column or raise error?
                    # SQL typically errors if ORDER BY column doesn't exist in SELECT list or FROM table
                    # For simplicity here, we can ignore or raise. Let's raise.
                    # However, if the column might exist in some records but not others, .get() handles it.
                    # The main issue is if the column name itself is wrong.
                    # _get_sortable_value handles r.get(col_name) which returns None if key missing.
                    pass # Let sort proceed, r.get(col_name) will be None.

                current_records.sort(key=lambda r: _get_sortable_value(r.get(col_name)), reverse=is_descending)
    
    # 3. Apply SELECT clause
    selected_column_names: List[str]
    if select_columns_str.strip() == "*":
        if not current_records:
            return []
        # Gather all unique column names from the (filtered and sorted) records
        all_cols_set = set()
        for r in current_records:
            all_cols_set.update(r.keys())
        selected_column_names = sorted(list(all_cols_set)) # Consistent order for '*'
        if not selected_column_names and current_records: # Records are list of empty dicts
             return [{} for _ in current_records]

    else:
        selected_column_names = [col.strip() for col in select_columns_str.split(',') if col.strip()]
        if not selected_column_names:
             raise ValueError("SELECT clause column list is empty or invalid.")


    return select_columns(current_records, selected_column_names)


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

    print("\n--- SQL Query Handler Examples ---")
    sql_dataset = [
        {'id': 1, 'name': 'Alice', 'age': 30, 'city': 'New York', 'active': True},
        {'id': 2, 'name': 'Bob', 'age': 24, 'occupation': 'Engineer', 'city': 'Paris', 'active': False},
        {'id': 3, 'name': 'Charlie', 'city': 'London', 'age': 35, 'active': True},
        {'id': 4, 'name': 'David', 'age': 24, 'city': 'New York', 'occupation': 'Artist', 'active': True},
        {'id': 5, 'name': 'Eve', 'age': 30, 'city': 'Paris', 'occupation': 'Engineer', 'active': False}
    ]
    print("\nOriginal SQL Dataset:")
    for row in sql_dataset:
        print(row)

    queries = [
        "SELECT name, age FROM dataset WHERE age > 25 ORDER BY age DESC, name ASC",
        "SELECT * FROM dataset WHERE city = 'New York' AND active = true",
        "SELECT id, name, city FROM dataset WHERE occupation = 'Engineer' OR city = 'London'",
        "SELECT name, occupation FROM dataset WHERE age < 30 ORDER BY name",
        "SELECT * FROM dataset WHERE name = 'NonExistent'",
        "SELECT city, age FROM dataset ORDER BY city ASC, age DESC",
        "SELECT name FROM dataset WHERE age = 24", # Test numeric equality
        "SELECT name FROM dataset WHERE active = false", # Test boolean equality
        "SELECT name, age, city FROM dataset WHERE city != 'Paris' ORDER BY age ASC",
        "SELECT * FROM dataset" # Select all
    ]

    for i, query_str in enumerate(queries):
        print(f"\nQuery {i+1}: {query_str}")
        try:
            result = handle_sql_query(sql_dataset, query_str)
            if result:
                for row in result:
                    print(row)
            else:
                print("No records found.")
        except ValueError as e:
            print(f"Error: {e}")

    print("\n--- SQL Query Handler Edge Case Examples ---")
    # Malformed queries
    malformed_queries = [
        "SELECT name WHERE age > 25", # Missing FROM (though our parser makes it optional)
        "SELECT name, age FROM dataset WHERE age >", # Incomplete condition
        "SELECT name, age FROM dataset ORDER BY age NONEXISTENT_ORDER", # Invalid order direction
        "SELECT FROM dataset", # Missing columns
        "SELECT name age FROM dataset", # Missing comma
        "SELECT name FROM dataset WHERE city = Paris", # Unquoted string literal
    ]
    for i, query_str in enumerate(malformed_queries):
        print(f"\nMalformed Query {i+1}: {query_str}")
        try:
            result = handle_sql_query(sql_dataset, query_str)
            for row in result: print(row)
        except ValueError as e:
            print(f"Error: {e}")

    # Empty dataset
    print("\nQuery on empty dataset: SELECT name FROM dataset WHERE age > 20")
    try:
        result = handle_sql_query([], "SELECT name FROM dataset WHERE age > 20")
        if result:
            for row in result: print(row)
        else:
            print("No records found (empty dataset).")
    except ValueError as e:
        print(f"Error: {e}")
