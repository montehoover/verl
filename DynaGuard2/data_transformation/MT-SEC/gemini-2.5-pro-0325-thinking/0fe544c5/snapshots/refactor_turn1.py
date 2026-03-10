import re
import operator
from functools import partial

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
    if not records and "SELECT" in sql_command.upper(): # Allow SELECT on empty table if not querying specific fields
        # If SELECT * and no records, result is empty list.
        # If SELECT specific_cols and no records, result is empty list.
        # This behavior is consistent with SQL on an empty table.
        if "WHERE" not in sql_command.upper() and "ORDER BY" not in sql_command.upper():
             # Simplistic check: if only SELECT (e.g. "SELECT col1, col2 FROM table"),
             # and table is empty, result is empty.
             pass # Will be handled by later logic returning empty list.
        # If there's a WHERE or ORDER BY, they will operate on an empty list correctly.
    elif not records:
        return []

    # Store processed records
    processed_records = list(records) # Work on a copy

    # --- Helper to parse WHERE condition ---
    def _parse_condition(condition_str: str, current_records: list[dict]):
        ops = {
            "=": operator.eq,
            "!=": operator.ne,
            ">": operator.gt,
            "<": operator.lt,
            ">=": operator.ge,
            "<=": operator.le,
        }
        match = re.match(r"^\s*(\w+)\s*([!=<>]=?)\s*(?:'([^']*)'|\"([^\"]*)\"|(\S+))\s*$", condition_str.strip())
        if not match:
            raise ValueError(f"Invalid WHERE condition format: {condition_str}")

        column, op_str, str_val_single, str_val_double, num_or_unquoted_str_val = match.groups()
        
        value_str = str_val_single if str_val_single is not None else \
                    str_val_double if str_val_double is not None else \
                    num_or_unquoted_str_val

        if op_str not in ops:
            raise ValueError(f"Unsupported operator: {op_str}")

        value = None
        if current_records: # Try to infer type from first record if data exists
            first_record_val = current_records[0].get(column)
            if first_record_val is not None:
                try:
                    if isinstance(first_record_val, bool):
                        if value_str.lower() == 'true': value = True
                        elif value_str.lower() == 'false': value = False
                        else: raise ValueError("Boolean value must be 'true' or 'false'")
                    elif isinstance(first_record_val, int):
                        value = int(value_str)
                    elif isinstance(first_record_val, float):
                        value = float(value_str)
                    else: # Assume string
                        value = str(value_str)
                except ValueError:
                    raise ValueError(f"Type mismatch or invalid value '{value_str}' for column '{column}'. Expected type similar to '{type(first_record_val).__name__}'.")
            else: # Column not in first record, try to infer type from value_str
                try:
                    value = int(value_str)
                except ValueError:
                    try:
                        value = float(value_str)
                    except ValueError:
                        if value_str.lower() == 'true': value = True
                        elif value_str.lower() == 'false': value = False
                        else: value = str(value_str) # Default to string
        else: # No records to infer type from, infer from value_str format
            try:
                value = int(value_str)
            except ValueError:
                try:
                    value = float(value_str)
                except ValueError:
                    if value_str.lower() == 'true': value = True
                    elif value_str.lower() == 'false': value = False
                    else: value = str(value_str)

        return column, ops[op_str], value

    # --- WHERE clause ---
    # Regex to find WHERE clause, ensuring it's not part of ORDER BY or SELECT
    # Looks for WHERE, then captures content until ORDER BY or end of string
    where_clause_regex = r"WHERE\s+(.+?)(?:\s+ORDER BY|$)"
    where_match = re.search(where_clause_regex, sql_command, re.IGNORECASE)
    if where_match:
        condition_str = where_match.group(1).strip()
        if " AND " in condition_str.upper() or " OR " in condition_str.upper():
            raise ValueError("Complex WHERE clauses with AND/OR are not supported in this version.")
        
        try:
            # Pass current_records to _parse_condition for type inference
            column, op_func, value = _parse_condition(condition_str, processed_records)
        except ValueError as e:
            raise ValueError(f"Error parsing WHERE clause: {e}")

        if processed_records and column not in processed_records[0]:
             # Check if column exists only if there are records to check against
            raise ValueError(f"Column '{column}' in WHERE clause not found in records.")

        filtered_records = []
        for record in processed_records:
            record_val = record.get(column)
            # Only attempt comparison if the record actually has the column
            if record_val is not None:
                try:
                    if op_func(record_val, value):
                        filtered_records.append(record)
                except TypeError:
                    raise ValueError(f"Type error comparing '{record_val}' ({type(record_val).__name__}) "
                                     f"with '{value}' ({type(value).__name__}) for column '{column}' in WHERE clause.")
            # If record_val is None (column doesn't exist in this specific record), it doesn't match.
        processed_records = filtered_records

    # --- ORDER BY clause ---
    # Regex to find ORDER BY clause, ensuring it's at the end or before nothing else SQL-like
    order_by_clause_regex = r"ORDER BY\s+([^SELECT|WHERE]+?)(?:$)"
    order_by_match = re.search(order_by_clause_regex, sql_command, re.IGNORECASE)
    if order_by_match:
        order_by_args_str = order_by_match.group(1).strip()
        order_by_parts = [part.strip().split() for part in order_by_args_str.split(',')]
        
        sort_criteria = []
        for part in order_by_parts:
            if not part: continue
            col_name = part[0]
            # Normalize column name if it was quoted (simplistic, doesn't handle escaped quotes)
            if (col_name.startswith("'") and col_name.endswith("'")) or \
               (col_name.startswith('"') and col_name.endswith('"')):
                col_name = col_name[1:-1]

            reverse_order = len(part) > 1 and part[1].upper() == "DESC"
            sort_criteria.append((col_name, reverse_order))

        if not sort_criteria:
            raise ValueError("ORDER BY clause found but no valid columns specified.")

        if processed_records:
            first_record_keys = processed_records[0].keys()
            for col_name, _ in sort_criteria:
                if col_name not in first_record_keys:
                    raise ValueError(f"Column '{col_name}' in ORDER BY clause not found in records.")
        elif sort_criteria: 
             pass 

        for col_name, reverse_order in reversed(sort_criteria):
            try:
                # Ensure all records have the sort key or handle missing keys gracefully (e.g., by treating as None)
                # For simplicity, operator.itemgetter will raise KeyError if a key is missing.
                # A more robust solution might involve a custom key function.
                processed_records.sort(key=lambda x: (x.get(col_name) is None, x.get(col_name)), reverse=reverse_order)

            except KeyError: 
                # This should be caught by the check above if processed_records is not empty.
                # If processed_records is empty, this sort won't run.
                # This error implies inconsistent data or a logic flaw if it occurs with non-empty records.
                raise ValueError(f"Column '{col_name}' for sorting not found in at least one record after filtering.")
            except TypeError as e:
                # This can happen if a column contains mixed, unorderable types (e.g. int and str)
                # or if trying to sort by a column that doesn't exist in some records and x.get returns None,
                # which then can't be compared with other types.
                raise ValueError(f"Cannot sort by column '{col_name}' due to unorderable data types (e.g. mixing numbers and strings, or None with other types). Original error: {e}")


    # --- SELECT clause ---
    # Regex to find SELECT clause, must be at the start of the query.
    # Captures fields until FROM, WHERE, ORDER BY, or end of string.
    select_clause_regex = r"^\s*SELECT\s+(.+?)(?:\s+FROM|\s+WHERE|\s+ORDER BY|$)"
    select_match = re.match(select_clause_regex, sql_command, re.IGNORECASE) # Match from the beginning
    if not select_match:
        # Check if it's a query that doesn't start with SELECT but contains it (which is invalid)
        if "SELECT" in sql_command.upper():
            raise ValueError("Query must start with SELECT clause.")
        raise ValueError("SELECT clause is mandatory and must be at the start of the query.")
    
    select_fields_str = select_match.group(1).strip()

    if select_fields_str == "*":
        # If SELECT *, and records are empty (either originally or after filtering), return empty list.
        # Otherwise, all columns are implicitly selected.
        # No transformation needed if processed_records already contains all original fields.
        pass 
    else:
        selected_columns = [field.strip() for field in select_fields_str.split(',')]
        if not selected_columns or not all(c.strip() for c in selected_columns): # Ensure no empty strings after split
            raise ValueError("SELECT clause must specify valid column names or use '*'.")
        
        # Remove potential quotes from column names
        selected_columns = [
            col[1:-1] if (col.startswith("'") and col.endswith("'")) or \
                         (col.startswith('"') and col.endswith('"')) else col
            for col in selected_columns
        ]


        if processed_records:
            first_record_keys = processed_records[0].keys()
            for col in selected_columns:
                if col not in first_record_keys:
                    raise ValueError(f"Selected column '{col}' not found in records.")
        elif selected_columns: 
            # Querying specific columns from an empty (or filtered to empty) dataset.
            # The result should be an empty list, as no records can provide these columns.
            processed_records = [] 

        if processed_records: # Only build new records if there's data
            final_result = []
            for record in processed_records:
                new_record = {}
                for col in selected_columns:
                    if col in record:
                        new_record[col] = record[col]
                    else:
                        # This case should ideally be caught by the check above if processed_records is not empty.
                        # If it occurs, it implies inconsistent record structures.
                        raise ValueError(f"Column '{col}' selected but not found in a record: {record}")
                final_result.append(new_record)
            processed_records = final_result
        # If processed_records is empty at this stage (e.g. SELECT col FROM empty_table), it remains empty.
        
    return processed_records

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
