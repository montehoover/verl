import re
import operator
# functools.partial is not used in this implementation, so it's omitted.

# Helper for mapping SQL-like operators to Python's operator functions
_OPERATORS = {
    '=': operator.eq,
    '!=': operator.ne,
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
}

def _parse_value(val_str: str):
    """
    Parses a value string from a SQL query.
    Handles quoted strings, integers, and floats.
    """
    val_str = val_str.strip()
    if (val_str.startswith("'") and val_str.endswith("'")) or \
       (val_str.startswith('"') and val_str.endswith('"')):
        return val_str[1:-1]  # Return as string
    try:
        return int(val_str)
    except ValueError:
        try:
            return float(val_str)
        except ValueError:
            raise ValueError(f"WHERE clause value '{val_str}' must be a number or a quoted string.")


def extract_fields(data: list[dict], fields: list[str], conditions: dict) -> list[dict]:
    """
    Extracts specified fields from a list of dictionaries and filters records based on conditions.

    Args:
        data: A list of dictionaries.
        fields: A list of field names to extract.
        conditions: A dictionary specifying field conditions to filter records.
                    Example: {"age": 30, "city": "New York"}

    Returns:
        A new list of dictionaries, where each dictionary contains only
        the specified fields from the original dictionaries that match all conditions.

    Raises:
        ValueError: If a condition field is not found in a record when checking conditions.
    """
    result = []
    for record in data:
        record_matches_conditions = True
        # Apply conditions if any are provided (i.e., if conditions dictionary is not empty)
        if conditions: 
            for cond_key, cond_value in conditions.items():
                if cond_key not in record:
                    raise ValueError(f"Condition field '{cond_key}' not found in record: {record}")
                if record[cond_key] != cond_value:
                    record_matches_conditions = False
                    break  # Stop checking other conditions for this record

        if record_matches_conditions:
            # If record matches all conditions (or if conditions was empty), extract fields
            new_record = {}
            for field in fields:
                if field in record:
                    new_record[field] = record[field]
            result.append(new_record) # Add the extracted record (could be empty if fields is empty or no specified fields found)
    return result

if __name__ == '__main__':
    # Example Usage
    sample_data = [
        {"id": 1, "name": "Alice", "age": 30, "city": "New York"},
        {"id": 2, "name": "Bob", "age": 24, "occupation": "Engineer"},
        {"id": 3, "name": "Charlie", "city": "London", "age": 35}
    ]

    # --- Original examples (adapted with empty conditions dictionary {}) ---
    print("--- Original behavior with empty conditions ---")
    fields_to_extract = ["name", "age"]
    # Pass empty dict for conditions to signify no filtering
    extracted_data = extract_fields(sample_data, fields_to_extract, {})
    print("Extracted data (name, age) with no filter:")
    for item in extracted_data:
        print(item)

    fields_to_extract_2 = ["id", "city", "occupation"]
    extracted_data_2 = extract_fields(sample_data, fields_to_extract_2, {})
    print("\nExtracted data (id, city, occupation) with no filter:")
    for item in extracted_data_2:
        print(item)

    fields_to_extract_3 = ["non_existent_field"]
    extracted_data_3 = extract_fields(sample_data, fields_to_extract_3, {})
    print("\nExtracted data (non_existent_field) with no filter (expect list of empty dicts):")
    for item in extracted_data_3:
        print(item) 

    fields_to_extract_4 = [] # Empty list of fields
    extracted_data_4 = extract_fields(sample_data, fields_to_extract_4, {})
    print("\nExtracted data (no fields) with no filter (expect list of empty dicts):")
    for item in extracted_data_4:
        print(item)

    empty_data = []
    extracted_data_empty = extract_fields(empty_data, fields_to_extract, {})
    print("\nExtracted data (empty input data) with no filter:")
    for item in extracted_data_empty:
        print(item)

    # --- New examples with filtering ---
    print("\n--- New examples with filtering ---")

    # Example 1: Filter by age
    conditions_age_30 = {"age": 30}
    extracted_filtered_1 = extract_fields(sample_data, ["name", "city"], conditions_age_30)
    print("\nFiltered by age=30, extract (name, city):")
    for item in extracted_filtered_1:
        print(item) # Expected: [{'name': 'Alice', 'city': 'New York'}]

    # Example 2: Filter by multiple fields
    conditions_bob = {"name": "Bob", "occupation": "Engineer"}
    extracted_filtered_2 = extract_fields(sample_data, ["id", "age"], conditions_bob)
    print("\nFiltered by name='Bob' AND occupation='Engineer', extract (id, age):")
    for item in extracted_filtered_2:
        print(item) # Expected: [{'id': 2, 'age': 24}]

    # Example 3: Filter resulting in no matches (condition value mismatch)
    conditions_no_match_value = {"city": "Paris"}
    extracted_filtered_3 = extract_fields(sample_data, ["name"], conditions_no_match_value)
    print("\nFiltered by city='Paris', extract (name):")
    for item in extracted_filtered_3:
        print(item) # Expected: []

    # Example 4: Filter with empty fields list
    extracted_filtered_4 = extract_fields(sample_data, [], conditions_age_30)
    print("\nFiltered by age=30, extract (no fields) (expect list with one empty dict):")
    for item in extracted_filtered_4:
        print(item) # Expected: [{}]

    # Example 5: Filter with condition on a non-existent field (ValueError)
    print("\nAttempting filter with condition on non-existent field 'country':")
    conditions_bad_field = {"country": "USA"}
    try:
        extract_fields(sample_data, ["name"], conditions_bad_field)
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")

    # Example 6: Filter with conditions that don't match any record's value for a field
    conditions_age_100 = {"age": 100}
    extracted_filtered_5 = extract_fields(sample_data, ["name"], conditions_age_100)
    print("\nFiltered by age=100, extract (name):")
    for item in extracted_filtered_5:
        print(item) # Expected: []

    # Example 7: All records match condition (empty condition dict), 
    # but some fields to extract are not present in all matching records
    conditions_all_match_implicit = {} 
    fields_mixed_presence = ["name", "occupation"]
    extracted_filtered_6 = extract_fields(sample_data, fields_mixed_presence, conditions_all_match_implicit)
    print("\nNo filter, extract (name, occupation):")
    for item in extracted_filtered_6:
        print(item)

# --- run_sql_query function and examples ---

def run_sql_query(records: list[dict], command: str) -> list[dict]:
    """
    Executes a SQL-like query on a list of dictionaries.
    Supports SELECT, WHERE, and ORDER BY clauses.

    Args:
        records: A list of dictionaries to query.
        command: A SQL-like query string.
                 Example: "SELECT name, age FROM table WHERE age > 30 ORDER BY name ASC"
                          (Note: "FROM table" is illustrative; the table is `records`)

    Returns:
        A list of dictionaries representing the query result.

    Raises:
        ValueError: If the query is malformed, a field is not found,
                    or an operation fails.
    """
    original_command = command # For error messages
    command_upper = command.upper()

    # Find clause indices
    select_idx = command_upper.find("SELECT ")
    where_idx = command_upper.find(" WHERE ")
    orderby_idx = command_upper.find(" ORDER BY ")

    if select_idx == -1:
        raise ValueError(f"Query must start with SELECT: '{original_command}'")

    # Determine clause boundaries
    fields_start = select_idx + len("SELECT ")
    
    # End of fields_str is start of WHERE, or start of ORDER BY, or end of command
    fields_end = len(command)
    if where_idx != -1:
        fields_end = where_idx
    elif orderby_idx != -1: # No WHERE, but ORDER BY exists
        fields_end = orderby_idx
    
    fields_str = command[fields_start:fields_end].strip()

    # Optional FROM clause removal (it's ignored anyway)
    from_clause_match = re.search(r"\s+FROM\s+\w+", fields_str, re.IGNORECASE)
    if from_clause_match:
        fields_str = fields_str[:from_clause_match.start()].strip()

    if not fields_str:
        raise ValueError(f"No fields specified in SELECT clause: '{original_command}'")

    where_str = None
    if where_idx != -1:
        where_start = where_idx + len(" WHERE ")
        where_end = orderby_idx if orderby_idx != -1 and orderby_idx > where_idx else len(command)
        where_str = command[where_start:where_end].strip()
        if not where_str:
            raise ValueError(f"Empty WHERE clause: '{original_command}'")

    orderby_str = None
    if orderby_idx != -1:
        orderby_start = orderby_idx + len(" ORDER BY ")
        orderby_str = command[orderby_start:].strip()
        if not orderby_str:
            raise ValueError(f"Empty ORDER BY clause: '{original_command}'")

    # --- Start processing records ---
    processed_records = list(records) # Work on a copy

    # 1. WHERE clause
    if where_str:
        # Simple WHERE: field op value. No AND/OR support for this version.
        # Example: "age > 30" or "city = 'New York'"
        where_match = re.match(r"(\w+)\s*([<>=!]+)\s*(.*)", where_str, re.IGNORECASE)
        if not where_match:
            raise ValueError(f"Malformed WHERE clause: '{where_str}' in query '{original_command}'")
        
        cond_field, cond_op_str, cond_val_str = where_match.groups()
        cond_field = cond_field.strip()
        cond_op_str = cond_op_str.strip()
        
        if cond_op_str not in _OPERATORS:
            raise ValueError(f"Unsupported operator in WHERE clause: '{cond_op_str}' in query '{original_command}'")
        cond_op_func = _OPERATORS[cond_op_str]
        
        try:
            cond_val_parsed = _parse_value(cond_val_str.strip())
        except ValueError as e: # Raised by _parse_value
            raise ValueError(f"Invalid value in WHERE clause ('{cond_val_str}'): {e} in query '{original_command}'")


        filtered_records = []
        for record in processed_records:
            if cond_field not in record:
                raise ValueError(f"Field '{cond_field}' in WHERE clause not found in record: {record} (Query: '{original_command}')")
            
            record_val = record[cond_field]
            matches = False
            try:
                # Type coercion for comparison
                if isinstance(cond_val_parsed, str):
                    matches = cond_op_func(str(record_val), cond_val_parsed)
                elif isinstance(cond_val_parsed, int):
                    matches = cond_op_func(int(record_val), cond_val_parsed)
                elif isinstance(cond_val_parsed, float):
                    matches = cond_op_func(float(record_val), cond_val_parsed)
                else: # Should not happen due to _parse_value
                    matches = cond_op_func(record_val, cond_val_parsed)
            except (ValueError, TypeError):
                # ValueError from int()/float() conversion, TypeError from comparison of incompatible types
                matches = False # Treat as non-match
            
            if matches:
                filtered_records.append(record)
        processed_records = filtered_records

    # 2. ORDER BY clause
    if orderby_str:
        orderby_parts = orderby_str.strip().split()
        orderby_field = orderby_parts[0]
        orderby_direction = "ASC"
        if len(orderby_parts) > 1:
            direction_token = orderby_parts[1].upper()
            if direction_token in ["ASC", "DESC"]:
                orderby_direction = direction_token
            else:
                raise ValueError(f"Invalid ORDER BY direction: '{orderby_parts[1]}' in query '{original_command}'")

        reverse_sort = (orderby_direction == "DESC")
        
        def sort_key_func(record):
            val = record.get(orderby_field) # Use .get() for graceful handling of missing keys
            if val is None:
                return (0, None) # Sort None values first (or last if reverse_sort and primary key is 0)
            if isinstance(val, (int, float)):
                return (1, val)
            if isinstance(val, str):
                return (2, val)
            return (3, str(val)) # Fallback for other types

        try:
            processed_records.sort(key=sort_key_func, reverse=reverse_sort)
        except TypeError: 
            # Should be rare with the tuple-based key, but possible if internal comparisons fail
            raise ValueError(f"Error sorting by field '{orderby_field}': Incompatible data types for comparison in query '{original_command}'.")

    # 3. SELECT clause (Projection)
    selected_field_names = [f.strip() for f in fields_str.split(',') if f.strip()]
    
    if not selected_field_names:
        raise ValueError(f"No fields specified in SELECT clause: '{original_command}'")

    final_result = []
    if selected_field_names == ['*']:
        final_result = processed_records # Return all fields of the (filtered and sorted) records
    else:
        for record in processed_records:
            new_record = {}
            for field_name in selected_field_names:
                if field_name in record:
                    new_record[field_name] = record[field_name]
                else:
                    # As per typical SQL behavior, if a selected field doesn't exist,
                    # it's often returned as NULL. Here, we use None.
                    new_record[field_name] = None 
            final_result.append(new_record)
            
    return final_result


if __name__ == '__main__':
    # (Previous examples for extract_fields remain above)
    # ...

    print("\n\n--- run_sql_query Examples ---")
    
    # Re-use sample_data from extract_fields examples
    # sample_data = [
    #     {"id": 1, "name": "Alice", "age": 30, "city": "New York"},
    #     {"id": 2, "name": "Bob", "age": 24, "occupation": "Engineer"},
    #     {"id": 3, "name": "Charlie", "city": "London", "age": 35}
    # ]

    print("\nQuery 1: SELECT name, age WHERE age > 25 ORDER BY age DESC")
    query1_result = run_sql_query(sample_data, "SELECT name, age WHERE age > 25 ORDER BY age DESC")
    for row in query1_result:
        print(row)
    # Expected:
    # {'name': 'Charlie', 'age': 35}
    # {'name': 'Alice', 'age': 30}

    print("\nQuery 2: SELECT * WHERE city = 'New York'")
    query2_result = run_sql_query(sample_data, "SELECT * WHERE city = 'New York'")
    for row in query2_result:
        print(row)
    # Expected:
    # {'id': 1, 'name': 'Alice', 'age': 30, 'city': 'New York'}

    print("\nQuery 3: SELECT name, occupation ORDER BY name ASC")
    # Note: 'occupation' is missing for Alice and Charlie. It will be None.
    query3_result = run_sql_query(sample_data, "SELECT name, occupation FROM DUMMY_TABLE ORDER BY name ASC") # FROM is ignored
    for row in query3_result:
        print(row)
    # Expected:
    # {'name': 'Alice', 'occupation': None}
    # {'name': 'Bob', 'occupation': 'Engineer'}
    # {'name': 'Charlie', 'occupation': None}

    print("\nQuery 4: SELECT id WHERE name = 'NonExistent'")
    query4_result = run_sql_query(sample_data, "SELECT id WHERE name = 'NonExistent'")
    for row in query4_result:
        print(row)
    # Expected: (empty list)

    print("\nQuery 5: Malformed query (missing SELECT)")
    try:
        run_sql_query(sample_data, "name, age WHERE age > 25")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\nQuery 6: Condition on non-existent field")
    try:
        run_sql_query(sample_data, "SELECT name WHERE country = 'USA'")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\nQuery 7: SELECT non_existent_field, name")
    query7_result = run_sql_query(sample_data, "SELECT non_existent_field, name")
    for row in query7_result:
        print(row)
    # Expected:
    # {'non_existent_field': None, 'name': 'Alice'}
    # {'non_existent_field': None, 'name': 'Bob'}
    # {'non_existent_field': None, 'name': 'Charlie'}
    
    print("\nQuery 8: SELECT name WHERE age = '30' (string value for numeric comparison)")
    query8_result = run_sql_query(sample_data, "SELECT name, age WHERE age = '30'")
    for row in query8_result:
        print(row)
    # Expected: {'name': 'Alice', 'age': 30}

    print("\nQuery 9: SELECT name WHERE name = Bob (unquoted string value in condition)")
    try:
        run_sql_query(sample_data, "SELECT name WHERE name = Bob")
    except ValueError as e:
        print(f"Caught expected error: {e}") # Error from _parse_value

    print("\nQuery 10: SELECT * ORDER BY non_existent_field")
    # This should sort, with non_existent_field effectively being None for all. Order might be stable.
    query10_result = run_sql_query(sample_data, "SELECT * ORDER BY non_existent_field ASC")
    for row in query10_result:
        print(row)
    # Expected: Original order, as all 'non_existent_field' values are None.
    # {'id': 1, 'name': 'Alice', 'age': 30, 'city': 'New York'}
    # {'id': 2, 'name': 'Bob', 'age': 24, 'occupation': 'Engineer'}
    # {'id': 3, 'name': 'Charlie', 'city': 'London', 'age': 35}

    print("\nQuery 11: SELECT name, age WHERE id >= 2 ORDER BY age ASC")
    query11_result = run_sql_query(sample_data, "SELECT name, age WHERE id >= 2 ORDER BY age ASC")
    for row in query11_result:
        print(row)
    # Expected:
    # {'name': 'Bob', 'age': 24}
    # {'name': 'Charlie', 'age': 35}
