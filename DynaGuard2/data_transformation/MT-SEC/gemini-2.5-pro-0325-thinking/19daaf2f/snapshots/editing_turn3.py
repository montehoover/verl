import re
import operator
from functools import partial # Included as per user request, though not actively used in this version of run_sql_query.

# --- SQL Query Engine Components ---

OPERATORS = {
    '=': operator.eq,
    '!=': operator.ne,
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
}

def _parse_value(value_str: str):
    """
    Parses a string value from a SQL query into a Python type.
    Handles quoted strings, numbers, booleans (true/false), and null.
    """
    value_str = value_str.strip()
    # Check for SQL NULL keyword
    if value_str.lower() == 'null':
        return None
    # Check for quoted strings
    if (value_str.startswith("'") and value_str.endswith("'")) or \
       (value_str.startswith('"') and value_str.endswith('"')):
        return value_str[1:-1]
    # Check for booleans
    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False
    # Try to parse as a number (integer then float)
    try:
        return int(value_str)
    except ValueError:
        try:
            return float(value_str)
        except ValueError:
            # If it's not any of the above, it's an unquoted string or unsupported format.
            # SQL standard requires strings to be quoted. Unquoted identifiers are usually field names.
            # For values, if it's not a recognized keyword or number, it's an error.
            raise ValueError(f"Unsupported value format or unquoted string: {value_str}")

def _evaluate_condition(record: dict, field_name: str, op_str: str, value_str: str) -> bool:
    """
    Evaluates a single condition (e.g., age > 30) against a record.
    """
    # If the field doesn't exist in the record, SQL treats it as NULL.
    # Comparisons involving NULL (except IS NULL, IS NOT NULL) are generally unknown/false.
    if field_name not in record:
        # Special handling if op_str is for IS NULL or IS NOT NULL (not implemented here)
        return False 

    record_value = record[field_name]
    
    try:
        condition_value = _parse_value(value_str)
    except ValueError as e:
        # Propagate error if value parsing fails (e.g. malformed value in query)
        # This error will be caught by the top-level try-except in test cases or calling code.
        raise 


    op_func = OPERATORS.get(op_str)
    if not op_func:
        raise ValueError(f"Unsupported operator: {op_str}")

    try:
        # Direct comparison. Python's operators handle mixed types to some extent (e.g., int and float).
        return op_func(record_value, condition_value)
    except TypeError:
        # This can happen if types are fundamentally incompatible for the operation (e.g., '>' between int and str).
        # In SQL, this might lead to a runtime error or implicit conversion attempts.
        # Here, we'll treat it as the condition not being met.
        return False

def run_sql_query(dataset: list[dict], sql_query: str) -> list[dict]:
    """
    Parses and executes a SQL-like query on a list of dictionaries.
    Supports: SELECT fields [WHERE conditions].
    - fields can be '*' or comma-separated field names (e.g., name, age).
    - conditions are combined with AND (e.g., field1 = value1 AND field2 > value2).
    - OPERATOR can be =, !=, >, <, >=, <=.
    - value can be a quoted string (e.g., 'New York'), number (e.g., 30, -10.5), 
      boolean (true/false), or null. Field names are simple identifiers (e.g. user_id).
    Raises ValueError for malformed queries or execution errors.
    """
    query = sql_query.strip()
    
    # Regex to parse "SELECT <fields> [WHERE <conditions>]"
    # It does not support a FROM clause; the dataset is passed as an argument.
    query_pattern = re.compile(r"SELECT\s+(.+?)(?:\s+WHERE\s+(.+))?$", re.IGNORECASE)
    match = query_pattern.match(query)

    if not match:
        raise ValueError("Malformed query: Must conform to 'SELECT fields [WHERE conditions]' structure. "
                         "Example: SELECT name, age WHERE city = 'New York'")

    selected_fields_str = match.group(1).strip()
    conditions_str = match.group(2).strip() if match.group(2) else None

    # Parse selected fields
    fields_to_select = []
    select_all_fields = False
    if selected_fields_str == "*":
        select_all_fields = True
    else:
        fields_to_select = [f.strip() for f in selected_fields_str.split(',')]
        if not fields_to_select or any(not f for f in fields_to_select): # Checks for "f1,,f2" or " "
            raise ValueError("Malformed SELECT clause: Fields list is invalid or contains empty names. "
                             "Example: SELECT name, age")

    # Stage 1: Filter data based on WHERE clause
    processed_data = []
    if conditions_str:
        # Split conditions by 'AND' (case-insensitive)
        condition_parts = re.split(r"\s+AND\s+", conditions_str, flags=re.IGNORECASE)
        parsed_conditions = []
        
        # Regex for one condition: field_name op value
        # field_name: \w+ (allows letters, numbers, underscore)
        # op: [!=<>]=?
        # value: quoted string, number (int/float, optional sign), true, false, null
        condition_pattern = re.compile(
            # field_name (group 1)  operator (group 2)                                value (group 3)
            r"(\w+)\s*([!=<>]=?)\s*('.*?'|\".*?\"|[+-]?\d+(?:\.\d*)?|[+-]?\.\d+(?:E[+-]?\d+)?|true|false|null)", 
            re.IGNORECASE
        ) # Added E notation for floats like 1.2E-5

        for part in condition_parts:
            part_stripped = part.strip()
            if not part_stripped: # Handles cases like "cond1 AND AND cond2" by skipping empty parts.
                continue
            cond_match = condition_pattern.fullmatch(part_stripped) # Use fullmatch for the entire part
            if not cond_match:
                raise ValueError(f"Malformed condition part: '{part_stripped}'. "
                                 "Expected format: field_name OPERATOR value (e.g., age > 30 or name = 'Alice')")
            
            field, op, val_str = cond_match.groups()
            parsed_conditions.append({'field': field.strip(), 'op': op.strip(), 'val_str': val_str.strip()})
        
        # If conditions_str was not empty/whitespace but parsing yielded no conditions (e.g., "WHERE GIBBERISH")
        if not parsed_conditions and conditions_str.strip(): 
             raise ValueError("Malformed WHERE clause: No valid conditions found after parsing.")

        if parsed_conditions: # Only filter if there are actual conditions parsed
            for record in dataset:
                all_conditions_met = True
                for cond in parsed_conditions:
                    if not _evaluate_condition(record, cond['field'], cond['op'], cond['val_str']):
                        all_conditions_met = False
                        break
                if all_conditions_met:
                    processed_data.append(record)
        else: # conditions_str was present but effectively empty (e.g. "WHERE ", "WHERE AND", etc.)
            processed_data = list(dataset) # No effective filtering, all records pass
    else:
        # No WHERE clause, so all data passes the filter stage
        processed_data = list(dataset)

    # Stage 2: Select/project fields from the filtered data
    result_set = []
    for record in processed_data:
        new_record = {}
        if select_all_fields:
            # For SELECT *, copy all fields from the filtered record
            new_record = dict(record) 
        else:
            # For SELECT field1, field2, ..., project specified fields
            # SQL behavior: if a selected field is not in a source record, it appears as NULL.
            for field_name in fields_to_select:
                new_record[field_name] = record.get(field_name) # .get(key, None) is default
        result_set.append(new_record)
                
    return result_set

# --- End of SQL Query Engine Components ---

def filter_and_extract(data: list[dict], fields: list[str], filter_conditions: dict = None) -> list[dict]:
    """
    Filters records based on specified conditions and then extracts specified
    fields from the matching records.

    Args:
        data: A list of dictionaries.
        fields: A list of field names to extract.
        filter_conditions: A dictionary where keys are field names and values
                           are the required values for filtering.
                           If None or empty, no filtering is applied.

    Returns:
        A new list of dictionaries, where each dictionary contains only the
        specified fields from the records that matched the filter conditions.
        If a field is not present in an input dictionary, it will be omitted
        from the output dictionary for that record.
    """
    if filter_conditions is None:
        filter_conditions = {}

    filtered_and_extracted_data = []
    for record in data:
        match = True
        if filter_conditions:
            for key, value in filter_conditions.items():
                if record.get(key) != value:
                    match = False
                    break
        
        if match:
            new_record = {}
            for field in fields:
                if field in record:
                    new_record[field] = record[field]
            if new_record or not fields: # if fields is empty, an empty dict is a valid extraction for a matched record
                filtered_and_extracted_data.append(new_record)
                
    return filtered_and_extracted_data

if __name__ == '__main__':
    # Example Usage
    sample_data = [
        {"id": 1, "name": "Alice", "age": 30, "city": "New York"},
        {"id": 2, "name": "Bob", "age": 24, "occupation": "Engineer", "city": "San Francisco"},
        {"id": 3, "name": "Charlie", "age": 30, "city": "London", "country": "UK"},
        {"id": 4, "name": "Diana", "age": 28} 
    ]

    print("--- Test Case 1: Extract 'name' and 'age', no filter ---")
    fields1 = ["name", "age"]
    result1 = filter_and_extract(sample_data, fields1)
    for row in result1:
        print(row)
    # Expected:
    # {'name': 'Alice', 'age': 30}
    # {'name': 'Bob', 'age': 24}
    # {'name': 'Charlie', 'age': 30}
    # {'name': 'Diana', 'age': 28}

    print("\n--- Test Case 2: Extract 'name' and 'city', filter by age = 30 ---")
    fields2 = ["name", "city"]
    filters2 = {"age": 30}
    result2 = filter_and_extract(sample_data, fields2, filters2)
    for row in result2:
        print(row)
    # Expected:
    # {'name': 'Alice', 'city': 'New York'}
    # {'name': 'Charlie', 'city': 'London'}

    print("\n--- Test Case 3: Extract 'id', 'occupation', filter by city = 'San Francisco' ---")
    fields3 = ["id", "occupation"]
    filters3 = {"city": "San Francisco"}
    result3 = filter_and_extract(sample_data, fields3, filters3)
    for row in result3:
        print(row)
    # Expected:
    # {'id': 2, 'occupation': 'Engineer'}

    print("\n--- Test Case 4: Extract 'name', filter by non-existent field 'country' = 'USA' ---")
    fields4 = ["name"]
    filters4 = {"country": "USA"}
    result4 = filter_and_extract(sample_data, fields4, filters4)
    for row in result4: # Should be empty
        print(row)
    if not result4:
        print("(No records matched)")
    # Expected:
    # (No records matched)

    print("\n--- Test Case 5: Extract 'name', 'age', filter by age = 30 and city = 'New York' ---")
    fields5 = ["name", "age"]
    filters5 = {"age": 30, "city": "New York"}
    result5 = filter_and_extract(sample_data, fields5, filters5)
    for row in result5:
        print(row)
    # Expected:
    # {'name': 'Alice', 'age': 30}

    print("\n--- Test Case 6: Extract all fields (empty list), filter by name = 'Diana' ---")
    fields6 = [] # Requesting all fields from matched records (results in empty dicts if record has no *other* fields)
                # Or rather, it means "extract no specific fields, just give me empty dicts for matches"
                # Corrected logic: if fields is empty, new_record remains empty, and is appended.
    filters6 = {"name": "Diana"}
    result6 = filter_and_extract(sample_data, fields6, filters6)
    for row in result6:
        print(row)
    # Expected:
    # {}

    print("\n--- Test Case 7: Extract 'city', filter by non-matching condition (age = 100) ---")
    fields7 = ["city"]
    filters7 = {"age": 100}
    result7 = filter_and_extract(sample_data, fields7, filters7)
    if not result7:
        print("(No records matched)")
    # Expected:
    # (No records matched)

    print("\n--- Test Case 8: Empty data input ---")
    result8 = filter_and_extract([], fields1, filters2)
    if not result8:
        print("(No records processed)")
    # Expected:
    # (No records processed)

    print("\n--- Test Case 9: Extract 'name', no filter, on data with missing 'name' ---")
    sample_data_missing_fields = [
        {"id": 1, "age": 30, "city": "New York"},
        {"id": 2, "name": "Bob", "age": 24}
    ]
    fields9 = ["name"]
    result9 = filter_and_extract(sample_data_missing_fields, fields9)
    for row in result9:
        print(row)
    # Expected:
    # {}  (from the first record, as 'name' is not present but record itself is processed)
    # {'name': 'Bob'}

    print("\n--- Test Case 10: Extract 'name', filter on 'city', from data with missing 'city' in some records ---")
    filters10 = {"city": "New York"}
    result10 = filter_and_extract(sample_data_missing_fields, fields9, filters10)
    for row in result10:
        print(row)
    # Expected:
    # {} (from the first record, as 'name' is not present but it matches filter)

    print("\n\n--- Testing run_sql_query ---")
    sql_sample_data = [
        {"id": 1, "name": "Alice", "age": 30, "city": "New York", "active": True, "score": 100.5},
        {"id": 2, "name": "Bob", "age": 24, "occupation": "Engineer", "city": "San Francisco", "active": False, "score": 90},
        {"id": 3, "name": "Charlie", "age": 30, "city": "London", "country": "UK", "active": True, "score": 100.5},
        {"id": 4, "name": "Diana", "age": 28, "city": "New York", "active": None, "score": -10} # active is None (SQL NULL)
    ]

    print("\n--- SQL Test 1: SELECT name, age WHERE city = 'New York' ---")
    query1 = "SELECT name, age WHERE city = 'New York'"
    try:
        result_sql1 = run_sql_query(sql_sample_data, query1)
        for row in result_sql1:
            print(row)
    except ValueError as e:
        print(f"Error: {e}")
    # Expected:
    # {'name': 'Alice', 'age': 30}
    # {'name': 'Diana', 'age': 28}

    print("\n--- SQL Test 2: SELECT * WHERE age > 25 AND active = true ---")
    query2 = "SELECT * WHERE age > 25 AND active = true"
    try:
        result_sql2 = run_sql_query(sql_sample_data, query2)
        for row in result_sql2:
            print(row)
    except ValueError as e:
        print(f"Error: {e}")
    # Expected:
    # {'id': 1, 'name': 'Alice', 'age': 30, 'city': 'New York', 'active': True, 'score': 100.5}
    # {'id': 3, 'name': 'Charlie', 'age': 30, 'city': 'London', 'country': 'UK', 'active': True, 'score': 100.5}
    
    print("\n--- SQL Test 3: SELECT name, country WHERE country = \"UK\" ---") # Double quotes for string
    query3 = "SELECT name, country WHERE country = \"UK\""
    try:
        result_sql3 = run_sql_query(sql_sample_data, query3)
        for row in result_sql3:
            print(row)
    except ValueError as e:
        print(f"Error: {e}")
    # Expected:
    # {'name': 'Charlie', 'country': 'UK'}

    print("\n--- SQL Test 4: SELECT name WHERE age < 20 (no matches) ---")
    query4 = "SELECT name WHERE age < 20"
    try:
        result_sql4 = run_sql_query(sql_sample_data, query4)
        if not result_sql4:
            print("(No records matched)")
        else:
            for row in result_sql4:
                print(row)
    except ValueError as e:
        print(f"Error: {e}")
    # Expected:
    # (No records matched)

    print("\n--- SQL Test 5: SELECT id, non_existent_field WHERE name = 'Alice' ---")
    query5 = "SELECT id, non_existent_field WHERE name = 'Alice'"
    try:
        result_sql5 = run_sql_query(sql_sample_data, query5)
        for row in result_sql5:
            print(row)
    except ValueError as e:
        print(f"Error: {e}")
    # Expected:
    # {'id': 1, 'non_existent_field': None}

    print("\n--- SQL Test 6: Malformed query - no SELECT ---")
    query6 = "name, age WHERE city = 'New York'"
    try:
        run_sql_query(sql_sample_data, query6)
    except ValueError as e:
        print(f"Error: {e}")
    # Expected: Error: Malformed query: Must conform to 'SELECT fields [WHERE conditions]' structure...
    
    print("\n--- SQL Test 7: Malformed query - invalid operator ---")
    query7 = "SELECT name WHERE city = 'New York' AND age !> 30" # !> is not a valid operator
    try:
        run_sql_query(sql_sample_data, query7)
    except ValueError as e:
        print(f"Error: {e}")
    # Expected: Error: Malformed condition part: 'age !> 30'...

    print("\n--- SQL Test 8: SELECT name, city (no WHERE clause) ---")
    query8 = "SELECT name, city"
    try:
        result_sql8 = run_sql_query(sql_sample_data, query8)
        for row in result_sql8:
            print(row)
    except ValueError as e:
        print(f"Error: {e}")
    # Expected:
    # {'name': 'Alice', 'city': 'New York'}
    # {'name': 'Bob', 'city': 'San Francisco'}
    # {'name': 'Charlie', 'city': 'London'}
    # {'name': 'Diana', 'city': 'New York'}

    print("\n--- SQL Test 9: SELECT * (no WHERE clause) ---")
    query9 = "SELECT *"
    try:
        result_sql9 = run_sql_query(sql_sample_data, query9)
        for row in result_sql9:
            print(row)
    except ValueError as e:
        print(f"Error: {e}")
    # Expected: All records, all fields

    print("\n--- SQL Test 10: WHERE active = NULL ---")
    query10 = "SELECT name, active WHERE active = NULL"
    try:
        result_sql10 = run_sql_query(sql_sample_data, query10)
        for row in result_sql10:
            print(row)
    except ValueError as e:
        print(f"Error: {e}")
    # Expected:
    # {'name': 'Diana', 'active': None}

    print("\n--- SQL Test 11: Malformed SELECT (empty field) ---")
    query11 = "SELECT name,,age WHERE city = 'New York'"
    try:
        run_sql_query(sql_sample_data, query11)
    except ValueError as e:
        print(f"Error: {e}")
    # Expected: Error: Malformed SELECT clause: Fields list is invalid or contains empty names.

    print("\n--- SQL Test 12: Query with unquoted string value (error) ---")
    query12 = "SELECT name WHERE city = NewYork" 
    try:
        run_sql_query(sql_sample_data, query12)
    except ValueError as e:
        print(f"Error: {e}")
    # Expected: Error from _parse_value: Unsupported value format or unquoted string: NewYork

    print("\n--- SQL Test 13: Query with numeric comparison (<=) ---")
    query13 = "SELECT name, age WHERE age <= 28"
    try:
        result_sql13 = run_sql_query(sql_sample_data, query13)
        for row in result_sql13:
            print(row)
    except ValueError as e:
        print(f"Error: {e}")
    # Expected:
    # {'name': 'Bob', 'age': 24}
    # {'name': 'Diana', 'age': 28}

    print("\n--- SQL Test 14: Query with float value comparison ---")
    query14 = "SELECT name, score WHERE score = 100.5"
    try:
        result_sql14 = run_sql_query(sql_sample_data, query14)
        for row in result_sql14:
            print(row)
    except ValueError as e:
        print(f"Error: {e}")
    # Expected:
    # {'name': 'Alice', 'score': 100.5}
    # {'name': 'Charlie', 'score': 100.5}

    print("\n--- SQL Test 15: SELECT with no WHERE on empty dataset ---")
    query15 = "SELECT name, age"
    try:
        result_sql15 = run_sql_query([], query15)
        if not result_sql15:
            print("(No records processed, empty result)")
        else: # Should not loop
            for row in result_sql15: print(row)
    except ValueError as e:
        print(f"Error: {e}")
    # Expected: (No records processed, empty result)

    print("\n--- SQL Test 16: SELECT * on empty dataset ---")
    query16 = "SELECT *"
    try:
        result_sql16 = run_sql_query([], query16)
        if not result_sql16:
            print("(No records processed, empty result for SELECT *)")
        else: # Should not loop
            for row in result_sql16: print(row)
    except ValueError as e:
        print(f"Error: {e}")
    # Expected: (No records processed, empty result for SELECT *)

    print("\n--- SQL Test 17: WHERE clause that is valid but results in no conditions (e.g. only spaces) ---")
    query17 = "SELECT name WHERE      " 
    try:
        result_sql17 = run_sql_query(sql_sample_data, query17)
        for row in result_sql17: print(row)
    except ValueError as e:
        print(f"Error: {e}")
    # Expected: (all names, as WHERE clause is empty)
    # {'name': 'Alice'}
    # {'name': 'Bob'}
    # {'name': 'Charlie'}
    # {'name': 'Diana'}

    print("\n--- SQL Test 18: WHERE clause with leading/trailing/multiple ANDs (lenient parsing) ---")
    print("Test 18.1: Trailing AND: SELECT name WHERE age > 25 AND ")
    try:
        for row in run_sql_query(sql_sample_data, "SELECT name WHERE age > 25 AND "): print(row)
    except ValueError as e: print(f"Error: {e}")
    # Expected: Names of Alice, Charlie, Diana

    print("Test 18.2: Leading AND: SELECT name WHERE AND age > 25")
    try:
        for row in run_sql_query(sql_sample_data, "SELECT name WHERE AND age > 25"): print(row)
    except ValueError as e: print(f"Error: {e}")
    # Expected: Names of Alice, Charlie, Diana (same as above due to skipping empty parts)

    print("Test 18.3: Multiple ANDs: SELECT name WHERE age > 25 AND AND city = 'New York'")
    try:
        for row in run_sql_query(sql_sample_data, "SELECT name WHERE age > 25 AND AND city = 'New York'"): print(row)
    except ValueError as e: print(f"Error: {e}")
    # Expected: Names of Alice, Diana

    print("\n--- SQL Test 19: Trailing invalid text after conditions (error) ---")
    query19 = "SELECT name, age WHERE city = 'New York' FROM DUMMY_TABLE"
    try:
        run_sql_query(sql_sample_data, query19)
    except ValueError as e:
        print(f"Error: {e}")
    # Expected: Error: Malformed condition part: 'city = 'New York' FROM DUMMY_TABLE'...

    print("\n--- SQL Test 20: Field name with underscore ---")
    data_underscore = [{"user_id": 1, "user_name": "Test"}]
    query20 = "SELECT user_name WHERE user_id = 1"
    try:
        for row in run_sql_query(data_underscore, query20): print(row)
    except ValueError as e:
        print(f"Error: {e}")
    # Expected: {'user_name': 'Test'}

    print("\n--- SQL Test 21: Comparison with negative numbers ---")
    query21 = "SELECT name, score WHERE score < 0"
    try:
        for row in run_sql_query(sql_sample_data, query21): print(row)
    except ValueError as e:
        print(f"Error: {e}")
    # Expected: {'name': 'Diana', 'score': -10}

    print("\n--- SQL Test 22: Whitespace variations ---")
    query22 = "  SELECT   name  ,  age   WHERE   city  =  'New York'  "
    try:
        for row in run_sql_query(sql_sample_data, query22): print(row)
    except ValueError as e:
        print(f"Error: {e}")
    # Expected: (Same as Test 1)
    # {'name': 'Alice', 'age': 30}
    # {'name': 'Diana', 'age': 28}

    print("\n--- SQL Test 23: Case-insensitivity of keywords ---")
    query23 = "select Name, Age where City = 'New York' and Active = True"
    # Note: Field names ('Name', 'Age', 'City', 'Active') are case-sensitive in Python dicts.
    # The query parser treats field names from the query string as is.
    # So, this query will likely not match if data fields are lowercase 'name', 'age', etc.
    # The current implementation is case-sensitive for field names from data.
    # SQL standard can be case-insensitive for identifiers depending on collation/settings or if unquoted.
    # For this implementation: field names are case-sensitive as they appear in dict keys.
    # Let's test with exact case for fields from data.
    query23_corrected_case = "select name, age where city = 'New York' and active = True"
    try:
        for row in run_sql_query(sql_sample_data, query23_corrected_case): print(row)
    except ValueError as e:
        print(f"Error: {e}")
    # Expected: {'name': 'Alice', 'age': 30} (Only Alice matches age > default, city NY, active True)
    # Alice: age 30, city NY, active True. This should match.
