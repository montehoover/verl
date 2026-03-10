import re
import operator

# --- SQL Parsing Helper Functions ---

def _parse_select_columns(sql_command: str) -> list[str]:
    """Parses the SELECT clause and returns a list of column names or ['*']."""
    select_clause_regex = r"^\s*SELECT\s+(.+?)(?:\s+FROM|\s+WHERE|\s+ORDER BY|$)"
    select_match = re.match(select_clause_regex, sql_command, re.IGNORECASE)
    if not select_match:
        if "SELECT" in sql_command.upper():
            raise ValueError("Query must start with SELECT clause.")
        raise ValueError("SELECT clause is mandatory and must be at the start of the query.")

    select_fields_str = select_match.group(1).strip()
    if select_fields_str == "*":
        return ["*"]
    
    selected_columns = [field.strip() for field in select_fields_str.split(',')]
    if not selected_columns or not all(c.strip() for c in selected_columns):
        raise ValueError("SELECT clause must specify valid column names or use '*'.")

    # Remove potential quotes from column names
    return [
        col[1:-1] if (col.startswith("'") and col.endswith("'")) or \
                     (col.startswith('"') and col.endswith('"')) else col
        for col in selected_columns
    ]

def _parse_where_params(sql_command: str) -> tuple | None:
    """Parses the WHERE clause. Returns (column, op_str, value_str) or None."""
    where_clause_regex = r"WHERE\s+(.+?)(?:\s+ORDER BY|$)"
    where_match = re.search(where_clause_regex, sql_command, re.IGNORECASE)
    if not where_match:
        return None

    condition_str = where_match.group(1).strip()
    if " AND " in condition_str.upper() or " OR " in condition_str.upper():
        raise ValueError("Complex WHERE clauses with AND/OR are not supported in this version.")

    # Regex to capture column, operator, and value (string or number)
    # Handles simple strings in single or double quotes, and numbers/booleans
    condition_match = re.match(r"^\s*(\w+)\s*([!=<>]=?)\s*(?:'([^']*)'|\"([^\"]*)\"|(\S+))\s*$", condition_str.strip())
    if not condition_match:
        raise ValueError(f"Invalid WHERE condition format: {condition_str}")

    column, op_str, str_val_single, str_val_double, other_val = condition_match.groups()
    
    value_str = str_val_single if str_val_single is not None else \
                str_val_double if str_val_double is not None else \
                other_val
    
    supported_ops = ["=", "!=", ">", "<", ">=", "<="]
    if op_str not in supported_ops:
        raise ValueError(f"Unsupported operator: {op_str}")

    return column, op_str, value_str

def _parse_order_by_params(sql_command: str) -> list[tuple]:
    """Parses the ORDER BY clause. Returns list of (column_name, reverse_bool) tuples."""
    order_by_clause_regex = r"ORDER BY\s+([^SELECT|WHERE]+?)(?:$)" # Avoid capturing keywords from other clauses
    order_by_match = re.search(order_by_clause_regex, sql_command, re.IGNORECASE)
    if not order_by_match:
        return []

    order_by_args_str = order_by_match.group(1).strip()
    order_by_parts = [part.strip().split() for part in order_by_args_str.split(',')]
    
    sort_criteria = []
    for part in order_by_parts:
        if not part: continue
        col_name = part[0]
        # Normalize column name if it was quoted
        if (col_name.startswith("'") and col_name.endswith("'")) or \
           (col_name.startswith('"') and col_name.endswith('"')):
            col_name = col_name[1:-1]
        
        reverse_order = len(part) > 1 and part[1].upper() == "DESC"
        sort_criteria.append((col_name, reverse_order))

    if not sort_criteria and order_by_args_str: # ORDER BY was present but no valid columns
        raise ValueError("ORDER BY clause found but no valid columns specified.")
    return sort_criteria

# --- SQL Execution Helper Functions ---

def _execute_where_filter(
    records: list[dict], 
    column_name: str, 
    op_str: str, 
    value_str: str
) -> list[dict]:
    """Applies WHERE filter to records. Includes type inference for value_str."""
    if not records: # No records to filter
        return []

    # Type inference for value_str based on the first record's column type
    typed_value = None
    first_record_val_type_sample = records[0].get(column_name)

    if first_record_val_type_sample is not None:
        try:
            if isinstance(first_record_val_type_sample, bool):
                if value_str.lower() == 'true': typed_value = True
                elif value_str.lower() == 'false': typed_value = False
                else: raise ValueError("Boolean value must be 'true' or 'false'")
            elif isinstance(first_record_val_type_sample, int):
                typed_value = int(value_str)
            elif isinstance(first_record_val_type_sample, float):
                typed_value = float(value_str)
            else: # Assume string
                typed_value = str(value_str)
        except ValueError as e:
            raise ValueError(
                f"Type mismatch or invalid value '{value_str}' for column '{column_name}'. "
                f"Expected type similar to '{type(first_record_val_type_sample).__name__}'. Error: {e}"
            )
    else: # Column not in the first record (or first record doesn't exist, though caught above)
          # Attempt to infer type from value_str format alone
        try:
            typed_value = int(value_str)
        except ValueError:
            try:
                typed_value = float(value_str)
            except ValueError:
                if value_str.lower() == 'true': typed_value = True
                elif value_str.lower() == 'false': typed_value = False
                else: typed_value = str(value_str) # Default to string

    ops_map = {
        "=": operator.eq, "!=": operator.ne, ">": operator.gt,
        "<": operator.lt, ">=": operator.ge, "<=": operator.le,
    }
    op_func = ops_map[op_str]

    # Check if column exists in the first record (schema check)
    if column_name not in records[0]:
        raise ValueError(f"Column '{column_name}' in WHERE clause not found in records.")

    filtered_records = []
    for record in records:
        record_val = record.get(column_name)
        if record_val is not None: # Only compare if record has the column and it's not None
            try:
                if op_func(record_val, typed_value):
                    filtered_records.append(record)
            except TypeError:
                raise ValueError(
                    f"Type error comparing '{record_val}' ({type(record_val).__name__}) "
                    f"with '{typed_value}' ({type(typed_value).__name__}) for column '{column_name}'."
                )
    return filtered_records

def _execute_order_by_sort(records: list[dict], order_by_params: list[tuple]) -> list[dict]:
    """Sorts records based on order_by_params."""
    if not records or not order_by_params:
        return records

    # Check if all order by columns exist in the first record (schema check)
    first_record_keys = records[0].keys()
    for col_name, _ in order_by_params:
        if col_name not in first_record_keys:
            raise ValueError(f"Column '{col_name}' in ORDER BY clause not found in records.")

    # Apply sorting criteria in reverse order of specification for stable sort
    # (innermost sort key applied first)
    processed_records = list(records) # Work on a copy for sorting
    for col_name, reverse_order in reversed(order_by_params):
        try:
            # Sort key handles records where the sort column might be missing (treats as None)
            # and ensures None values are handled consistently (typically sorted to one end).
            processed_records.sort(key=lambda x: (x.get(col_name) is None, x.get(col_name)), reverse=reverse_order)
        except TypeError as e: # Should be rare with the (is None, val) tuple but good for safety
            raise ValueError(
                f"Cannot sort by column '{col_name}' due to unorderable data types "
                f"(e.g., mixing numbers and strings, or None with other types). Original error: {e}"
            )
    return processed_records

def _execute_select_projection(records: list[dict], select_columns: list[str]) -> list[dict]:
    """Projects records to include only selected_columns."""
    if not select_columns: # Should be caught by parser, but as a safeguard
        raise ValueError("No columns selected.")

    if select_columns == ["*"]:
        return records # Return all columns

    # Validate selected columns against the first record (if records exist)
    if records:
        first_record_keys = records[0].keys()
        for col in select_columns:
            if col not in first_record_keys:
                raise ValueError(f"Selected column '{col}' not found in records.")
    elif select_columns != ["*"]: # Specific columns selected but no records
        return []


    final_result = []
    for record in records:
        new_record = {col: record[col] for col in select_columns if col in record}
        # This check ensures all selected columns were present in each record.
        # It's somewhat redundant if the initial check against first_record_keys passed
        # and all records have a consistent schema.
        if len(new_record) != len(select_columns):
            missing_cols = [col for col in select_columns if col not in record]
            if records and missing_cols : # only raise if there were records to begin with
                 raise ValueError(f"Column(s) '{', '.join(missing_cols)}' selected but not found in a record: {record}")
        final_result.append(new_record)
    return final_result

def handle_sql_query(records: list[dict], sql_command: str) -> list[dict]:
    """
    Processes a custom SQL-like query on data represented as a list of dictionaries.
    Handles SELECT, WHERE, and ORDER BY clauses.

    Args:
        records: A list of dictionaries where each dictionary represents a record.
        sql_command: A string containing the SQL-like query.

    Returns:
        A list of dictionaries representing the query results.

    Raises:
        ValueError: If the query is malformed or execution fails.
    """
    # Initial check for empty records
    # If records are empty, WHERE and ORDER BY will correctly process an empty list.
    # SELECT will also correctly process an empty list.
    if not records:
        # Parse SELECT to see if it's valid, even on empty records.
        # This ensures "SELECT foo FROM bar" on an empty `bar` returns [] not an error,
        # but "SELCT foo FROM bar" (typo) is still an error.
        select_cols = _parse_select_columns(sql_command) # Validate SELECT clause syntax
        if select_cols != ["*"] and not records: # Specific columns from empty table
             return []
        # If SELECT *, and no records, result is empty list.
        # If WHERE or ORDER BY are present, they operate on empty list and return empty list.
        return []


    # Make a copy to work on
    current_records = list(records)

    # --- Pipeline Stage 1: Parse SQL Query ---
    # Note: Parsing functions raise ValueError on syntax errors.
    select_columns = _parse_select_columns(sql_command)
    where_params = _parse_where_params(sql_command) # (column, op_str, value_str) or None
    order_by_params = _parse_order_by_params(sql_command) # list of (column, reverse_bool)

    # --- Pipeline Stage 2: Execute WHERE clause ---
    if where_params:
        column_name, op_str, value_str = where_params
        current_records = _execute_where_filter(current_records, column_name, op_str, value_str)

    # --- Pipeline Stage 3: Execute ORDER BY clause ---
    if order_by_params:
        current_records = _execute_order_by_sort(current_records, order_by_params)

    # --- Pipeline Stage 4: Execute SELECT clause ---
    # _execute_select_projection handles the case of select_columns == ["*"]
    # and also validates columns if current_records is not empty.
    current_records = _execute_select_projection(current_records, select_columns)
        
    return current_records

if __name__ == '__main__':
    # Example Usage:
    sample_records = [
        {'id': 1, 'name': 'Alice', 'age': 30, 'city': 'New York', 'active': True},
        {'id': 2, 'name': 'Bob', 'age': 24, 'city': 'Los Angeles', 'active': False},
        {'id': 3, 'name': 'Charlie', 'age': 30, 'city': 'Chicago', 'active': True},
        {'id': 4, 'name': 'David', 'age': 28, 'city': 'New York', 'active': None}, # None for active
        {'id': 5, 'name': 'Eve', 'age': 22, 'city': 'Chicago', 'active': True},
        {'id': 6, 'name': 'Fiona', 'age': 35, 'city': 'Los Angeles', 'active': False},
        {'id': 7, 'name': 'George', 'age': 30, 'city': 'New York'}, # Missing 'active'
    ]

    # Test cases
    queries = [
        "SELECT name, age FROM records WHERE city = 'New York' ORDER BY age DESC",
        "SELECT * FROM records WHERE age > 25 ORDER BY name",
        "SELECT id, city FROM records WHERE age < 25",
        "SELECT name FROM records ORDER BY city ASC, name DESC",
        "SELECT * FROM records WHERE name = 'NonExistent'",
        "SELECT * FROM records",
        "SELECT name, age FROM records WHERE age = 30 ORDER BY name ASC",
        "SELECT name, active FROM records WHERE active = true ORDER BY name",
        "SELECT name, active FROM records WHERE active = false ORDER BY name",
        "SELECT name, age FROM records WHERE age >= 30",
        "SELECT name, age FROM records WHERE city != 'Chicago' ORDER BY age",
        "SELECT name, age FROM records WHERE name = 'Alice'",
        "SELECT name FROM records WHERE city = 'New York' ORDER BY name",
        "SELECT name, city FROM records ORDER BY age", # Sort by a field not selected
        "SELECT name FROM records WHERE age = 30 ORDER BY city DESC, name ASC",
        "   SELECT   name,    age   FROM    records   WHERE   city  =  'New York'   ORDER   BY   age   DESC  ", # Test with extra spaces
        "select name, age from records where city = 'New York' order by age desc", # Test with lowercase keywords
        "SELECT \"name\", 'age' FROM records WHERE \"city\" = 'New York' ORDER BY 'age' DESC", # Quoted identifiers
    ]

    for query_idx, query in enumerate(queries):
        print(f"Test Case {query_idx + 1}: Query: {query}")
        try:
            # Pass a fresh copy of sample_records for each query
            result = handle_sql_query(list(map(dict, sample_records)), query)
            if result:
                for row_idx, row in enumerate(result):
                    print(f"  Row {row_idx + 1}: {row}")
            else:
                print("  Result: [] (No records found or selected)")
        except ValueError as e:
            print(f"  Error: {e}")
        print("-" * 30)

    # Test with empty records
    print("Query on empty records: SELECT name FROM records WHERE age > 30")
    try:
        result = handle_sql_query([], "SELECT name FROM records WHERE age > 30")
        print(f"  Result: {result}")
    except ValueError as e:
        print(f"  Error: {e}")
    print("-" * 30)
    
    print("Query on empty records: SELECT * FROM records")
    try:
        result = handle_sql_query([], "SELECT * FROM records")
        print(f"  Result: {result}")
    except ValueError as e:
        print(f"  Error: {e}")
    print("-" * 30)

    print("Query on empty records: SELECT name, age FROM records")
    try:
        result = handle_sql_query([], "SELECT name, age FROM records")
        print(f"  Result: {result}")
    except ValueError as e:
        print(f"  Error: {e}")
    print("-" * 30)


    # Error case examples
    error_queries = [
        "SELECT name FROM records WHERE age ! 30", # Invalid operator
        "SELECT name FROM records WHERE city = NewYork", # Unquoted string value
        "SELECT non_existent_col FROM records",
        "SELECT name FROM records ORDER BY non_existent_col",
        "SELECT name FROM records WHERE non_existent_col = 10",
        "SELECT name, age FROM records WHERE age = 'thirty'", # Type mismatch for comparison (age is int)
        "INSERT INTO records VALUES (1)", # Unsupported operation / malformed select
        "SELECT name FROM records WHERE age > 'abc'", # Invalid comparison (age is int, 'abc' is not number)
        "SELECT name FROM", # Incomplete select
        "WHERE name = 'Alice'", # No SELECT
        "SELECT name WHERE name = 'Alice'", # Missing FROM (though FROM is implicit)
        "SELECT name, FROM records", # Badly formed select list
        "SELECT name FROM records WHERE age = ", # Incomplete WHERE
        "SELECT name FROM records ORDER BY", # Incomplete ORDER BY
        "SELECT name FROM records WHERE city = 'New York' AND age > 25", # AND not supported
        "SELECT name FROM records WHERE age > 20 OR city = 'Chicago'", # OR not supported
        "SELECT name FROM records WHERE age = true", # Comparing int age with boolean
    ]
    for query_idx, query in enumerate(error_queries):
        print(f"Error Test Case {query_idx + 1}: Query: {query}")
        try:
            result = handle_sql_query(list(map(dict, sample_records)), query)
            print(f"  Result (unexpected success): {result}")
        except ValueError as e:
            print(f"  Error (expected): {e}")
        print("-" * 30)
        
    # Test sorting with None values
    records_with_none = [
        {'name': 'A', 'value': 10},
        {'name': 'B', 'value': None},
        {'name': 'C', 'value': 5},
        {'name': 'D', 'value': None},
    ]
    print("Test sorting with None values (ASC): SELECT * FROM records_with_none ORDER BY value ASC")
    try:
        result = handle_sql_query(list(map(dict, records_with_none)), "SELECT * FROM records_with_none ORDER BY value ASC")
        for row in result: print(f"  {row}")
    except ValueError as e:
        print(f"  Error: {e}")
    print("-" * 30)

    print("Test sorting with None values (DESC): SELECT * FROM records_with_none ORDER BY value DESC")
    try:
        result = handle_sql_query(list(map(dict, records_with_none)), "SELECT * FROM records_with_none ORDER BY value DESC")
        for row in result: print(f"  {row}")
    except ValueError as e:
        print(f"  Error: {e}")
    print("-" * 30)

    # Test WHERE clause with a column not present in all records
    records_missing_col = [
        {'id': 1, 'name': 'Alice', 'age': 30},
        {'id': 2, 'name': 'Bob', 'city': 'LA'}, # No 'age'
    ]
    print("Test WHERE with column missing in some records: SELECT * FROM records_missing_col WHERE age > 25")
    try:
        result = handle_sql_query(list(map(dict, records_missing_col)), "SELECT * FROM records_missing_col WHERE age > 25")
        for row in result: print(f"  {row}") # Should only return Alice
    except ValueError as e:
        print(f"  Error: {e}")
    print("-" * 30)
