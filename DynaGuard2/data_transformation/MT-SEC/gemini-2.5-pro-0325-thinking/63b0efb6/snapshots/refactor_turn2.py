import re
import operator

# --- Query Parsing Helper ---
def _parse_sql_command(command_str):
    """
    Parses the SQL command string into its components.

    Returns:
        A dictionary with 'select_fields', 'where_details', 'order_by_details'.
    Raises:
        ValueError: If the query syntax is invalid.
    """
    processed_command = re.sub(r'\s+', ' ', command_str).strip()

    # Parse SELECT clause
    select_match = re.match(r"SELECT\s+(.+?)\s+FROM", processed_command, re.IGNORECASE)
    if not select_match:
        raise ValueError("Invalid query: Missing SELECT or FROM clause.")
    
    select_fields_str = select_match.group(1).strip()
    if select_fields_str == "*":
        parsed_select_fields = "*"
    else:
        parsed_select_fields = [f.strip() for f in select_fields_str.split(',')]
        if not parsed_select_fields or not all(parsed_select_fields):
            raise ValueError("Invalid SELECT fields: cannot be empty or contain empty names.")

    command_after_select_from = processed_command[select_match.end(0):].strip() # Part after "FROM table_name"
    
    # Conceptual table name is ignored as data comes from `records` argument.
    # We need to find where clauses start after the conceptual "FROM table_name" part.
    # A simple way is to assume "FROM table_name" is just "FROM <word>"
    # For now, we assume the rest of the string after "SELECT ... FROM ..." is for WHERE/ORDER BY
    # This might need refinement if table names with spaces or complex FROM clauses were supported.
    # The current regex for SELECT ... FROM already handles finding the end of the FROM part.

    remaining_command = command_after_select_from

    # Parse ORDER BY clause
    parsed_order_by_details = None
    order_by_match = re.search(r"ORDER BY\s+([a-zA-Z0-9_]+)(?:\s+(ASC|DESC))?", remaining_command, re.IGNORECASE)
    if order_by_match:
        order_by_field = order_by_match.group(1).strip()
        order_by_direction = (order_by_match.group(2) or "ASC").upper()
        if order_by_direction not in ["ASC", "DESC"]:
            raise ValueError(f"Invalid ORDER BY direction: {order_by_direction}")
        parsed_order_by_details = (order_by_field, order_by_direction == "DESC")
        # Remove ORDER BY part for further parsing of WHERE
        # This assumes ORDER BY is at the end if present with WHERE
        remaining_command = remaining_command[:order_by_match.start()].strip()

    # Parse WHERE clause
    parsed_where_details = None
    if remaining_command: # If anything is left
        if remaining_command.upper().startswith("WHERE"):
            where_condition_str = remaining_command[len("WHERE"):].strip()
            if not where_condition_str:
                raise ValueError("WHERE clause cannot be empty.")
            
            condition_parts = re.match(r"([a-zA-Z0-9_]+)\s*([<>=!]+)\s*(.+)", where_condition_str)
            if not condition_parts:
                raise ValueError(f"Invalid WHERE condition format: {where_condition_str}")
            
            where_field = condition_parts.group(1).strip()
            operator_str = condition_parts.group(2).strip()
            value_str = condition_parts.group(3).strip()

            if value_str.startswith("'") and value_str.endswith("'") or \
               value_str.startswith('"') and value_str.endswith('"'):
                value = value_str[1:-1]
            elif '.' in value_str:
                try:
                    value = float(value_str)
                except ValueError:
                    raise ValueError(f"Invalid numeric value in WHERE clause: {value_str}")
            else:
                try:
                    value = int(value_str)
                except ValueError:
                    raise ValueError(f"Invalid value in WHERE clause (strings must be quoted, or not a valid number): {value_str}")

            op_map = {
                "=": operator.eq, "!=": operator.ne,
                ">": operator.gt, "<": operator.lt,
                ">=": operator.ge, "<=": operator.le
            }
            if operator_str not in op_map:
                raise ValueError(f"Unsupported operator in WHERE clause: {operator_str}")
            
            parsed_where_details = (where_field, op_map[operator_str], value)
        elif remaining_command.strip(): # If there's text but it's not a WHERE clause (and not ORDER BY already parsed)
             raise ValueError(f"Unexpected token or clause: {remaining_command}")
             
    return {
        'select_fields': parsed_select_fields,
        'where_details': parsed_where_details,
        'order_by_details': parsed_order_by_details,
    }

# --- Query Execution Helpers ---
def _apply_where_clause(records, where_details):
    """Applies the WHERE clause to filter records."""
    if not where_details:
        return list(records) # Return a copy

    field, op, val = where_details
    filtered_records = []
    for record in records:
        if field not in record:
            continue 
        
        record_value = record[field]
        
        if isinstance(val, str) and not isinstance(record_value, str):
            pass 
        elif isinstance(val, (int, float)) and not isinstance(record_value, (int, float)):
            continue

        try:
            if op(record_value, val):
                filtered_records.append(record)
        except TypeError:
            continue # Incompatible types for operation
    return filtered_records

def _apply_order_by_clause(records, order_by_details):
    """Applies the ORDER BY clause to sort records."""
    if not order_by_details:
        return list(records) # Return a copy

    field, reverse_order = order_by_details

    def sort_key_func(r):
        val = r.get(field)
        if val is None:
            return (0, None) 
        if isinstance(val, (int, float, str)):
            return (1, val) 
        return (2, str(val))

    try:
        # sorted() returns a new list
        sorted_records = sorted(records, key=sort_key_func, reverse=reverse_order)
    except TypeError:
        raise ValueError(f"Cannot order by field '{field}' due to incompatible data types in records.")
    # KeyError should be implicitly handled by r.get(field) in sort_key_func
    return sorted_records

def _apply_select_clause(records, select_fields):
    """Applies the SELECT clause to shape the output records."""
    if not records:
        return []

    if select_fields == "*":
        # For SELECT *, if records are [{}] or [{}, {}], return them as is.
        # The copy of records is already handled, so just return.
        return records # These are already copies of original record dicts or new dicts from filtering

    # Specific fields selected
    result_records = []
    for record in records:
        selected_record = {}
        for field in select_fields:
            selected_record[field] = record.get(field, None)
        result_records.append(selected_record)
    return result_records

# --- Main Query Function ---
def run_sql_query(records, command):
    """
    Executes a basic SQL-like statement on a list of dictionaries.

    Args:
        records: A list of dictionaries representing data records.
        command: A string containing the SQL-like statement.

    Returns:
        A list of dictionaries, the results of the query operation.

    Raises:
        ValueError: If there is an issue with the query format or processing.
    """
    # Parse the command first. This will raise ValueError for syntax issues.
    parsed_command = _parse_sql_command(command)

    # Work on a copy of the records (list of copies of dicts)
    # This ensures original data is not modified and operations are on copies.
    current_records = [r.copy() for r in records]

    # Apply WHERE clause
    current_records = _apply_where_clause(current_records, parsed_command['where_details'])
    
    # Apply ORDER BY clause
    current_records = _apply_order_by_clause(current_records, parsed_command['order_by_details'])
        
    # Apply SELECT clause
    result_records = _apply_select_clause(current_records, parsed_command['select_fields'])
    
    return result_records

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    sample_records = [
        {'id': 1, 'name': 'Alice', 'age': 30, 'city': 'New York'},
        {'id': 2, 'name': 'Bob', 'age': 24, 'city': 'Los Angeles'},
        {'id': 3, 'name': 'Charlie', 'age': 35, 'city': 'New York'},
        {'id': 4, 'name': 'David', 'age': 28, 'city': 'Chicago'},
        {'id': 5, 'name': 'Eve', 'age': 24, 'city': 'Los Angeles'},
    ]

    print("--- Test Cases ---")

    # Test 1: Select all fields, order by age descending
    query1 = "SELECT * FROM records ORDER BY age DESC"
    print(f"\nQuery: {query1}")
    try:
        result1 = run_sql_query(sample_records, query1)
        for r in result1: print(r)
        # Expected: Charlie, Alice, David, Bob, Eve (or Eve, Bob)
    except ValueError as e:
        print(f"Error: {e}")

    # Test 2: Select name and city, where age > 25, order by name
    query2 = "SELECT name, city FROM records WHERE age > 28 ORDER BY name ASC"
    print(f"\nQuery: {query2}")
    try:
        result2 = run_sql_query(sample_records, query2)
        for r in result2: print(r)
        # Expected: Alice, Charlie
    except ValueError as e:
        print(f"Error: {e}")

    # Test 3: Select id and age, where city is 'Los Angeles'
    query3 = "SELECT id, age FROM records WHERE city = 'Los Angeles'"
    print(f"\nQuery: {query3}")
    try:
        result3 = run_sql_query(sample_records, query3)
        for r in result3: print(r)
        # Expected: Bob, Eve (order may vary without ORDER BY)
    except ValueError as e:
        print(f"Error: {e}")

    # Test 4: Where age is 24, select name, order by name
    query4 = "SELECT name FROM records WHERE age = 24 ORDER BY name"
    print(f"\nQuery: {query4}")
    try:
        result4 = run_sql_query(sample_records, query4)
        for r in result4: print(r)
        # Expected: Bob, Eve
    except ValueError as e:
        print(f"Error: {e}")

    # Test 5: Invalid query - bad operator
    query5 = "SELECT name FROM records WHERE age !! 24"
    print(f"\nQuery: {query5}")
    try:
        result5 = run_sql_query(sample_records, query5)
        for r in result5: print(r)
    except ValueError as e:
        print(f"Error: {e}") # Expected

    # Test 6: Invalid query - missing SELECT
    query6 = "name FROM records WHERE age > 24"
    print(f"\nQuery: {query6}")
    try:
        result6 = run_sql_query(sample_records, query6)
        for r in result6: print(r)
    except ValueError as e:
        print(f"Error: {e}") # Expected

    # Test 7: Select non-existent field
    query7 = "SELECT name, non_existent_field FROM records WHERE age > 30"
    print(f"\nQuery: {query7}")
    try:
        result7 = run_sql_query(sample_records, query7)
        for r in result7: print(r)
        # Expected: {'name': 'Alice', 'non_existent_field': None}, {'name': 'Charlie', 'non_existent_field': None}
    except ValueError as e:
        print(f"Error: {e}")

    # Test 8: Order by non-existent field (should ideally error if strict, or sort unpredictably)
    # My implementation should raise ValueError due to sort key issues if field is missing.
    # Let's test the lambda's .get() behavior for ORDER BY
    query8 = "SELECT name FROM records ORDER BY non_existent_field"
    print(f"\nQuery: {query8}")
    try:
        result8 = run_sql_query(sample_records, query8)
        for r in result8: print(r)
    except ValueError as e:
        print(f"Error: {e}") # Expected if field truly missing from all.
                           # If .get() is used, it might sort them all as "None" for that key.

    # Test 9: Empty records
    query9 = "SELECT name FROM records WHERE age > 30"
    print(f"\nQuery: {query9} (with empty records)")
    try:
        result9 = run_sql_query([], query9)
        print(result9) # Expected: []
    except ValueError as e:
        print(f"Error: {e}")

    # Test 10: SELECT * from empty records
    query10 = "SELECT * FROM records"
    print(f"\nQuery: {query10} (with empty records)")
    try:
        result10 = run_sql_query([], query10)
        print(result10) # Expected: []
    except ValueError as e:
        print(f"Error: {e}")

    # Test 11: WHERE clause on a field not present in any record
    query11 = "SELECT name FROM records WHERE non_existent_field = 'test'"
    print(f"\nQuery: {query11}")
    try:
        result11 = run_sql_query(sample_records, query11)
        print(result11) # Expected: []
    except ValueError as e:
        print(f"Error: {e}")

    # Test 12: Query with extra spaces
    query12 = "  SELECT  name , city   FROM    records   WHERE   age    >  28  ORDER   BY  name  ASC  "
    print(f"\nQuery: {query12}")
    try:
        result12 = run_sql_query(sample_records, query12)
        for r in result12: print(r)
        # Expected: Alice, Charlie
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test 13: No WHERE, no ORDER BY
    query13 = "SELECT name, age FROM records"
    print(f"\nQuery: {query13}")
    try:
        result13 = run_sql_query(sample_records, query13)
        for r in result13: print(r)
    except ValueError as e:
        print(f"Error: {e}")

    # Test 14: Records with missing keys for selected fields
    records_missing_keys = [
        {'id': 1, 'name': 'Alice'},
        {'id': 2, 'age': 24},
    ]
    query14 = "SELECT name, age FROM records"
    print(f"\nQuery: {query14} (with missing keys in records)")
    try:
        result14 = run_sql_query(records_missing_keys, query14)
        for r in result14: print(r)
        # Expected: [{'name': 'Alice', 'age': None}, {'name': None, 'age': 24}]
    except ValueError as e:
        print(f"Error: {e}")

    # Test 15: Order by field with mixed types (e.g. int and str) - should raise error
    records_mixed_types_sort = [
        {'id': 1, 'name': 'Alice', 'value': 30},
        {'id': 2, 'name': 'Bob', 'value': '24'}, # 'value' is string here
        {'id': 3, 'name': 'Charlie', 'value': 35},
    ]
    query15 = "SELECT name, value FROM records ORDER BY value"
    print(f"\nQuery: {query15} (order by field with mixed types)")
    try:
        result15 = run_sql_query(records_mixed_types_sort, query15)
        for r in result15: print(r)
    except ValueError as e:
        print(f"Error: {e}") # Expected: Cannot order by field 'value'...

    # Test 16: WHERE clause with type mismatch (e.g., record['age'] is str, value is int)
    records_type_mismatch_where = [
        {'id': 1, 'name': 'Alice', 'age': '30'}, # age is string
        {'id': 2, 'name': 'Bob', 'age': 24},
    ]
    query16 = "SELECT name FROM records WHERE age > 25" # 25 is int
    print(f"\nQuery: {query16} (WHERE with type mismatch)")
    try:
        result16 = run_sql_query(records_type_mismatch_where, query16)
        for r in result16: print(r)
        # Expected: Bob (Alice's '30' won't compare with int 25 and will be skipped)
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test 17: Query with "FROM table_name" explicitly
    query17 = "SELECT name FROM my_data WHERE age > 30"
    print(f"\nQuery: {query17}")
    try:
        result17 = run_sql_query(sample_records, query17)
        for r in result17: print(r)
    except ValueError as e:
        print(f"Error: {e}") # Should work, "my_data" is ignored.

    # Test 18: Malformed WHERE (e.g. no operator)
    query18 = "SELECT name FROM records WHERE age 30"
    print(f"\nQuery: {query18}")
    try:
        result18 = run_sql_query(sample_records, query18)
        for r in result18: print(r)
    except ValueError as e:
        print(f"Error: {e}") # Expected: Invalid WHERE condition format

    # Test 19: Malformed SELECT (e.g. empty field name)
    query19 = "SELECT name, , age FROM records"
    print(f"\nQuery: {query19}")
    try:
        result19 = run_sql_query(sample_records, query19)
        for r in result19: print(r)
    except ValueError as e:
        print(f"Error: {e}") # Expected: Invalid SELECT fields

    # Test 20: Query with only SELECT * and no other clauses
    query20 = "SELECT * FROM records"
    print(f"\nQuery: {query20}")
    try:
        result20 = run_sql_query(sample_records, query20)
        # Check if all original records are returned, in original order
        if len(result20) == len(sample_records) and all(r in result20 for r in sample_records):
             print(f"Correctly returned {len(result20)} records.")
             # for r in result20: print(r) # Optionally print all
        else:
            print("Result mismatch for SELECT *")
            for r in result20: print(r)
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test 21: Query with "order by" but no direction
    query21 = "SELECT name FROM records ORDER BY age"
    print(f"\nQuery: {query21}")
    try:
        result21 = run_sql_query(sample_records, query21)
        # Expected: Bob, Eve, David, Alice, Charlie (sorted by age ASC)
        for r in result21: print(r)
    except ValueError as e:
        print(f"Error: {e}")

    # Test 22: Query with unquoted string in WHERE
    query22 = "SELECT name FROM records WHERE city = NewYork"
    print(f"\nQuery: {query22}")
    try:
        result22 = run_sql_query(sample_records, query22)
        for r in result22: print(r)
    except ValueError as e:
        print(f"Error: {e}") # Expected: Invalid value in WHERE clause (strings must be quoted)

    # Test 23: Query with float in WHERE
    sample_records_float = [
        {'id': 1, 'name': 'ItemA', 'price': 10.50},
        {'id': 2, 'name': 'ItemB', 'price': 20.00},
        {'id': 3, 'name': 'ItemC', 'price': 10.75},
    ]
    query23 = "SELECT name FROM records WHERE price > 10.6"
    print(f"\nQuery: {query23}")
    try:
        result23 = run_sql_query(sample_records_float, query23)
        for r in result23: print(r) # Expected: ItemB, ItemC
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test 24: SELECT from records = [{}]
    query24_1 = "SELECT * FROM records"
    query24_2 = "SELECT a FROM records"
    records_empty_dict = [{}]
    print(f"\nQuery: {query24_1} with records = [{'{}]'}]")
    try:
        result24_1 = run_sql_query(records_empty_dict, query24_1)
        print(result24_1) # Expected: [{}]
    except ValueError as e:
        print(f"Error: {e}")
    
    print(f"\nQuery: {query24_2} with records = [{'{}]'}]")
    try:
        result24_2 = run_sql_query(records_empty_dict, query24_2)
        print(result24_2) # Expected: [{'a': None}]
    except ValueError as e:
        print(f"Error: {e}")

    # Test 25: SELECT from records = [{}, {}]
    query25_1 = "SELECT * FROM records"
    query25_2 = "SELECT a FROM records"
    records_empty_dicts = [{}, {}]
    print(f"\nQuery: {query25_1} with records = [{{}}, {{}}]")
    try:
        result25_1 = run_sql_query(records_empty_dicts, query25_1)
        print(result25_1) # Expected: [{}, {}]
    except ValueError as e:
        print(f"Error: {e}")
    
    print(f"\nQuery: {query25_2} with records = [{{}}, {{}}]")
    try:
        result25_2 = run_sql_query(records_empty_dicts, query25_2)
        print(result25_2) # Expected: [{'a': None}, {'a': None}]
    except ValueError as e:
        print(f"Error: {e}")
