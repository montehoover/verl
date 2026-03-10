import re
import operator
from functools import partial

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
    if not records and "SELECT" in command.upper(): # Allow empty records if not selecting
        pass # Will be handled by select if fields are specified
    elif not records:
        return []


    original_records = [r.copy() for r in records] # Work on a copy

    # Normalize command: remove extra spaces and convert to uppercase for clause keywords
    processed_command = re.sub(r'\s+', ' ', command).strip()

    # --- Parse SELECT clause ---
    select_match = re.match(r"SELECT\s+(.+?)\s+FROM", processed_command, re.IGNORECASE)
    if not select_match:
        raise ValueError("Invalid query: Missing SELECT or FROM clause.")
    
    select_fields_str = select_match.group(1).strip()
    
    # Remove SELECT part for further parsing
    command_after_select = processed_command[select_match.end(0):].strip()
    # The "FROM table_name" part is conceptual as 'records' is the table.
    # We need to ensure "FROM" was present, which select_match does.
    # We can effectively ignore the table name part if present, or assume it's implicit.
    # For simplicity, we assume "FROM records_variable_name" is implied by the function signature.
    
    # --- Parse ORDER BY clause (before WHERE for easier regex) ---
    order_by_clause = None
    order_by_match = re.search(r"ORDER BY\s+([a-zA-Z0-9_]+)(?:\s+(ASC|DESC))?", command_after_select, re.IGNORECASE)
    if order_by_match:
        order_by_field = order_by_match.group(1).strip()
        order_by_direction = (order_by_match.group(2) or "ASC").upper()
        if order_by_direction not in ["ASC", "DESC"]:
            raise ValueError(f"Invalid ORDER BY direction: {order_by_direction}")
        order_by_clause = (order_by_field, order_by_direction == "DESC")
        # Remove ORDER BY part for further parsing
        command_after_select = command_after_select[:order_by_match.start()].strip()

    # --- Parse WHERE clause ---
    where_clause = None
    if command_after_select: # If anything is left after FROM and ORDER BY
        where_match = re.match(r"WHERE\s+(.+)", command_after_select, re.IGNORECASE)
        if where_match:
            where_condition_str = where_match.group(1).strip()
            # Basic condition parsing: field operator value
            # Example: "age > 30", "name = 'Alice'"
            condition_parts = re.match(r"([a-zA-Z0-9_]+)\s*([<>=!]+)\s*(.+)", where_condition_str)
            if not condition_parts:
                raise ValueError(f"Invalid WHERE condition format: {where_condition_str}")
            
            where_field = condition_parts.group(1).strip()
            operator_str = condition_parts.group(2).strip()
            value_str = condition_parts.group(3).strip()

            # Convert value to appropriate type (number or string)
            if value_str.startswith("'") and value_str.endswith("'") or \
               value_str.startswith('"') and value_str.endswith('"'):
                value = value_str[1:-1] # String value
            elif '.' in value_str:
                try:
                    value = float(value_str)
                except ValueError:
                    raise ValueError(f"Invalid numeric value in WHERE clause: {value_str}")
            else:
                try:
                    value = int(value_str)
                except ValueError:
                    # Could be an unquoted string, which we might want to disallow or handle
                    raise ValueError(f"Invalid value in WHERE clause (strings must be quoted): {value_str}")

            op_map = {
                "=": operator.eq, "!=": operator.ne,
                ">": operator.gt, "<": operator.lt,
                ">=": operator.ge, "<=": operator.le
            }
            if operator_str not in op_map:
                raise ValueError(f"Unsupported operator in WHERE clause: {operator_str}")
            
            where_clause = (where_field, op_map[operator_str], value)
        elif command_after_select.strip(): # If there's text but it's not a WHERE clause
             raise ValueError(f"Unexpected token or clause after FROM: {command_after_select}")


    # --- Apply WHERE clause ---
    filtered_records = []
    if where_clause:
        field, op, val = where_clause
        for record in original_records:
            if field not in record:
                # Depending on strictness, could raise error or skip record
                # For now, skip if field missing for condition, or treat as not matching
                # To be SQL-like, a missing field in a WHERE condition often means the condition is false.
                continue 
            
            record_value = record[field]
            # Type consistency check for comparison (e.g. don't compare string '5' with int 30)
            if isinstance(val, str) and not isinstance(record_value, str):
                 # try to cast record_value to string for comparison, or skip/error
                 pass # Let comparison fail or succeed based on Python's behavior
            elif isinstance(val, (int, float)) and not isinstance(record_value, (int, float)):
                # If record_value is string '123', it won't match int 123 directly.
                # SQL might try to cast, but we'll be stricter here.
                continue # Skip if types are incompatible for numeric comparison

            try:
                if op(record_value, val):
                    filtered_records.append(record)
            except TypeError:
                # This can happen if types are incompatible for the operator (e.g., int > str)
                # Silently treat as non-match, or raise a more specific error
                continue
    else:
        filtered_records = original_records

    # --- Apply ORDER BY clause ---
    if order_by_clause:
        field, reverse_order = order_by_clause
        # Check if all records to be sorted have the order_by_field
        # and handle potential mixed types if any.
        # For simplicity, assume field exists and types are comparable.
        # A robust solution would check this or handle exceptions during sort.
        try:
            # Use a lambda that can handle missing keys gracefully for sorting
            # Records without the sort key will be placed according to default sort behavior (often first)
            # or could be explicitly handled (e.g., treated as None).
            def sort_key_func(r):
                val = r.get(field)
                if val is None: # Handle None values (e.g. make them smallest or largest)
                    # Python 3 sorts None less than other types by default.
                    # This behavior might be acceptable.
                    return (0, None) # Primary sort key for None presence, then None itself
                if isinstance(val, (int, float, str)):
                    return (1, val) # Primary sort key for type, then value
                return (2, str(val)) # Fallback for other types

            filtered_records.sort(key=sort_key_func, reverse=reverse_order)
        except TypeError:
            # This can happen if the field contains mixed, unorderable types (e.g., int and string)
            raise ValueError(f"Cannot order by field '{field}' due to incompatible data types in records.")
        except KeyError:
            # This should ideally be caught by .get() in lambda, but as a safeguard:
            raise ValueError(f"Order by field '{field}' not found in all records.")


    # --- Apply SELECT clause ---
    result_records = []
    if not filtered_records and select_fields_str != "*": # No records after filtering
         # If specific fields are requested from an empty set, the result is an empty set.
         # If SELECT *, it's also an empty set.
        return []
    
    if not filtered_records and select_fields_str == "*":
        return []


    if select_fields_str == "*":
        if not filtered_records: # e.g. SELECT * FROM empty_table
             # If original records were empty, and no where clause, this is correct.
             # If filtering resulted in empty, also correct.
            return []
        # Check if all original records had at least one key, if records is not empty
        # This is mostly for the case of records = [{}]
        if all(not r for r in filtered_records) and any(filtered_records): # list of empty dicts
             return [{} for _ in filtered_records] # return list of empty dicts
        result_records = filtered_records
    else:
        fields_to_select = [f.strip() for f in select_fields_str.split(',')]
        if not fields_to_select or not all(fields_to_select):
            raise ValueError("Invalid SELECT fields: cannot be empty or contain empty names.")

        if not filtered_records and fields_to_select: # e.g. SELECT name FROM empty_table
            return []

        for record in filtered_records:
            selected_record = {}
            for field in fields_to_select:
                if field not in record:
                    # SQL behavior for selecting a non-existent column can vary.
                    # Some might return NULL, others error. We'll include it as None.
                    selected_record[field] = None 
                else:
                    selected_record[field] = record[field]
            result_records.append(selected_record)
            
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
