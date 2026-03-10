import re
import operator
from functools import partial

# --- Helper for execute_query_cmd: Operator mapping ---
_SQL_OPERATORS = {
    '=': operator.eq,
    '!=': operator.ne,
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
    # Add other operators like 'LIKE' if needed, though LIKE needs custom logic
}

def _parse_value(value_str):
    """Helper to parse string, int, or float values from query."""
    value_str = value_str.strip()
    if (value_str.startswith("'") and value_str.endswith("'")) or \
       (value_str.startswith('"') and value_str.endswith('"')):
        return value_str[1:-1]  # String literal
    try:
        return int(value_str)  # Integer
    except ValueError:
        try:
            return float(value_str)  # Float
        except ValueError:
            # Could be a boolean literal or unquoted string, treat as string by default if not number
            # For stricter parsing, raise ValueError here.
            # For this implementation, we'll assume if not quoted and not number, it's an error or needs specific handling.
            # However, the regex for WHERE condition should ideally ensure value is quoted string or number.
            # This path might be hit if regex is too loose or for future extensions (e.g. field_name = other_field_name)
            raise ValueError(f"Unsupported value format in WHERE clause: {value_str}")


def execute_query_cmd(dataset_list: list[dict], sql_query: str) -> list[dict]:
    """
    Executes a SQL-like query on a list of dictionaries.

    Supports:
    - SELECT field1, field2, ... or SELECT *
    - WHERE field_name OPERATOR value (e.g., age > 25, name = 'Alice')
      (OPERATOR can be =, !=, >, <, >=, <=)
    - ORDER BY field_name [ASC|DESC]

    Args:
        dataset_list: A list of dictionaries representing the data.
        sql_query: A string containing the SQL-like query.

    Returns:
        A list of dictionaries representing the query result.

    Raises:
        ValueError: If the query is malformed, a field is not found where expected,
                    or an operation is unsupported.
    """
    query = sql_query.strip()
    
    # Regex to parse SELECT, optional WHERE, optional ORDER BY
    # SELECT <fields> [FROM <ignored_table_name>] [WHERE <conditions>] [ORDER BY <field> [ASC|DESC]]
    # We ignore FROM as dataset is an argument.
    match = re.match(
        r"SELECT\s+(?P<fields>.*?)\s*"
        r"(?:FROM\s+\w+\s*)?"  # Optional FROM clause (ignored)
        r"(?:WHERE\s+(?P<where_clause>.*?)\s*)?"
        r"(?:ORDER BY\s+(?P<orderby_clause>.*?)\s*)?$",
        query,
        re.IGNORECASE
    )

    if not match:
        raise ValueError(f"Malformed SQL query: {sql_query}")

    query_parts = match.groupdict()
    select_fields_str = query_parts["fields"].strip()
    where_clause_str = query_parts["where_clause"].strip() if query_parts["where_clause"] else None
    orderby_clause_str = query_parts["orderby_clause"].strip() if query_parts["orderby_clause"] else None

    processed_data = list(dataset_list) # Work on a copy

    # 1. WHERE clause processing
    if where_clause_str:
        # Simple WHERE condition parser: field OPERATOR value
        # Example: "age > 25", "name = 'Alice'"
        # For simplicity, this parser handles one condition. Complex conditions (AND/OR) are not supported.
        where_match = re.match(r"(\w+)\s*([<>=!]+)\s*(.*)", where_clause_str, re.IGNORECASE)
        if not where_match:
            raise ValueError(f"Malformed WHERE clause: {where_clause_str}")
        
        field_name, op_str, value_literal = where_match.groups()
        field_name = field_name.strip()
        op_str = op_str.strip()
        
        try:
            condition_value = _parse_value(value_literal)
        except ValueError as e:
             raise ValueError(f"Error parsing value in WHERE clause ('{value_literal}'): {e}")


        op_func = _SQL_OPERATORS.get(op_str)
        if not op_func:
            raise ValueError(f"Unsupported operator in WHERE clause: {op_str}")

        filtered_data = []
        for record in processed_data:
            if field_name not in record:
                # Depending on SQL dialect, this could be false or an error.
                # We'll treat it as the condition being false.
                # If strictness is required: raise ValueError(f"Field '{field_name}' not found in record for WHERE clause: {record}")
                continue 
            
            record_value = record[field_name]
            
            # Type consistency check (basic)
            # If record_value is str, condition_value should ideally be str.
            # If record_value is number, condition_value should be number.
            # This can get complex with type coercion rules in SQL.
            # For now, we rely on Python's comparison behavior.
            try:
                if op_func(record_value, condition_value):
                    filtered_data.append(record)
            except TypeError:
                # This can happen if trying to compare incompatible types, e.g., int > str
                raise ValueError(
                    f"Type mismatch in WHERE condition for field '{field_name}'. "
                    f"Cannot compare record value '{record_value}' (type {type(record_value).__name__}) "
                    f"with condition value '{condition_value}' (type {type(condition_value).__name__})."
                )
        processed_data = filtered_data

    # 2. ORDER BY clause processing
    if orderby_clause_str:
        orderby_match = re.match(r"(\w+)\s*(ASC|DESC)?", orderby_clause_str, re.IGNORECASE)
        if not orderby_match:
            raise ValueError(f"Malformed ORDER BY clause: {orderby_clause_str}")
        
        sort_field, sort_order = orderby_match.groups()
        sort_field = sort_field.strip()
        is_descending = sort_order and sort_order.upper() == 'DESC'

        # Custom sort key to handle potential missing fields or mixed types gracefully
        def sort_key_func(record):
            value = record.get(sort_field)
            if value is None:
                # Define how None values are sorted (e.g., first or last)
                # Here, they are treated as "smallest"
                return (0, None) # Primary key for type, secondary for value
            if isinstance(value, (int, float)):
                return (1, value)
            if isinstance(value, str):
                return (2, value.lower()) # Case-insensitive string sort
            return (3, value) # Other types

        try:
            processed_data.sort(key=sort_key_func, reverse=is_descending)
        except TypeError as e: # Should be caught by sort_key_func types, but as a fallback
            raise ValueError(f"Cannot sort by field '{sort_field}' due to incompatible data types or missing field handling: {e}")


    # 3. SELECT clause processing
    if select_fields_str == "*":
        final_results = processed_data # Return all fields of filtered/sorted records
    else:
        fields_to_select = [f.strip() for f in select_fields_str.split(',')]
        if not fields_to_select or not all(fields_to_select): # check for empty strings after split
             raise ValueError(f"No fields specified in SELECT clause or empty field name found: '{select_fields_str}'")

        final_results = []
        for record in processed_data:
            new_record = {}
            for field in fields_to_select:
                if field in record:
                    new_record[field] = record[field]
                # else: field not in record, so it's omitted from new_record
            if new_record or not fields_to_select: # Add if new_record is not empty or if no specific fields were requested (edge case, should be caught by "No fields specified")
                final_results.append(new_record)
                
    return final_results


def extract_fields(data: list[dict], fields: list[str], conditions: dict = None) -> list[dict]:
    """
    Extracts specified fields from a list of dictionaries after filtering.

    Args:
        data: A list of dictionaries.
        fields: A list of field names to extract.
        conditions: A dictionary of conditions to filter records.
                    Only records that meet all conditions are processed.
                    Defaults to None (no filtering).

    Returns:
        A new list of dictionaries, where each dictionary contains only the
        specified fields from the corresponding, filtered dictionary in the input data.
        If a field is not present in an input dictionary, it will be omitted
        from the output dictionary for that record.

    Raises:
        ValueError: If a field specified in `conditions` is not present in a record
                    being evaluated.
    """
    result = []
    for record in data:
        # Apply filtering based on conditions
        if conditions:
            record_matches_all_conditions = True
            for cond_key, cond_value in conditions.items():
                if cond_key not in record:
                    raise ValueError(f"Condition field '{cond_key}' not found in record: {record}")
                if record[cond_key] != cond_value:
                    record_matches_all_conditions = False
                    break
            
            if not record_matches_all_conditions:
                continue  # Skip to the next record if conditions are not met

        # If record passes filters (or no filters), proceed with field extraction
        new_record = {}
        for field in fields:
            if field in record:
                new_record[field] = record[field]
        
        if new_record:  # Add record only if it's not empty after extraction
            result.append(new_record)
    return result

if __name__ == '__main__':
    # Example Usage
    sample_data = [
        {"id": 1, "name": "Alice", "age": 30, "city": "New York"},
        {"id": 2, "name": "Bob", "age": 24, "occupation": "Engineer"},
        {"id": 3, "name": "Charlie", "city": "London", "country": "UK"},
        {"id": 4, "name": "Diana"} # Record with no fields to extract if fields are e.g. ["age", "city"]
    ]

    fields_to_extract = ["name", "city"]
    extracted_data = extract_fields(sample_data, fields_to_extract)
    print("Original Data:")
    for item in sample_data:
        print(item)
    print("\nExtracted Data (fields: {}):".format(fields_to_extract))
    for item in extracted_data:
        print(item)

    fields_to_extract_2 = ["age", "occupation"]
    extracted_data_2 = extract_fields(sample_data, fields_to_extract_2)
    print("\nExtracted Data (fields: {}):".format(fields_to_extract_2))
    for item in extracted_data_2:
        print(item)

    fields_to_extract_3 = ["country"]
    extracted_data_3 = extract_fields(sample_data, fields_to_extract_3)
    print("\nExtracted Data (fields: {}):".format(fields_to_extract_3))
    for item in extracted_data_3:
        print(item)

    # Example with no matching fields in any record
    fields_to_extract_4 = ["non_existent_field"]
    extracted_data_4 = extract_fields(sample_data, fields_to_extract_4)
    print("\nExtracted Data (fields: {}):".format(fields_to_extract_4))
    for item in extracted_data_4:
        print(item)
    if not extracted_data_4:
        print("No data extracted as fields were not found.")

    print("\n--- Testing with filtering conditions ---")

    # Example with filtering: extract name and age for records where city is "New York"
    fields_to_extract_filtered = ["name", "age"]
    conditions_filter_1 = {"city": "New York"}
    print(f"\nExtracting fields {fields_to_extract_filtered} with conditions {conditions_filter_1}:")
    try:
        filtered_data_1 = extract_fields(sample_data, fields_to_extract_filtered, conditions_filter_1)
        for item in filtered_data_1:
            print(item) # Expected: {"name": "Alice", "age": 30}
    except ValueError as e:
        print(f"Error: {e}")

    # Example with filtering: extract name, city and occupation for records where age is 24
    fields_to_extract_filtered_2 = ["name", "city", "occupation"]
    conditions_filter_2 = {"age": 24}
    print(f"\nExtracting fields {fields_to_extract_filtered_2} with conditions {conditions_filter_2}:")
    try:
        filtered_data_2 = extract_fields(sample_data, fields_to_extract_filtered_2, conditions_filter_2)
        for item in filtered_data_2:
            # Expected: {"name": "Bob", "occupation": "Engineer"} (city is not in Bob's record, so not extracted)
            print(item) 
    except ValueError as e:
        print(f"Error: {e}")

    # Example with filtering: no records match the conditions
    conditions_filter_3 = {"city": "Paris"}
    # Using fields_to_extract from a previous example: ["name", "city"]
    print(f"\nExtracting fields {fields_to_extract} with conditions {conditions_filter_3}:")
    try:
        filtered_data_3 = extract_fields(sample_data, fields_to_extract, conditions_filter_3)
        if filtered_data_3:
            for item in filtered_data_3:
                print(item)
        else:
            print("No records matched the conditions.")
    except ValueError as e:
        print(f"Error: {e}")

    # Example with filtering: record matches conditions, but specified fields for extraction are not in the record
    fields_empty_extraction = ["height"] # Assuming 'height' is not a common field in sample_data
    conditions_alice = {"name": "Alice"}
    print(f"\nExtracting fields {fields_empty_extraction} with conditions {conditions_alice}:")
    try:
        filtered_data_empty = extract_fields(sample_data, fields_empty_extraction, conditions_alice)
        if filtered_data_empty:
            for item in filtered_data_empty:
                print(item)
        else:
            # This is expected if Alice's record is found but 'height' is not in it.
            print("Record matched conditions, but no specified fields found for extraction.") 
    except ValueError as e:
        print(f"Error: {e}")

    # Example that should raise ValueError due to condition field not in a record
    print("\nTesting ValueError for condition on non-existent field:")
    conditions_error = {"non_existent_key": "value"}
    fields_for_error_test = ["name"]
    try:
        print(f"Attempting to extract fields {fields_for_error_test} with conditions {conditions_error}:")
        # This call should raise ValueError because 'non_existent_key' is not in the first record.
        extract_fields(sample_data, fields_for_error_test, conditions_error)
        print("ValueError was not raised as expected.") # Should not be reached
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")

    # Example: conditions is None (should behave as original, filtering only by fields)
    print("\nTesting with conditions=None (should extract based on fields only):")
    # Using fields_to_extract from a previous example: ["name", "city"]
    extracted_data_no_conditions = extract_fields(sample_data, fields_to_extract, None)
    for item in extracted_data_no_conditions:
        print(item)

    # Example: conditions is an empty dict (should behave as original, filtering only by fields)
    print("\nTesting with conditions={{}} (should extract based on fields only):")
    extracted_data_empty_conditions = extract_fields(sample_data, fields_to_extract, {})
    for item in extracted_data_empty_conditions:
        print(item)

    print("\n\n--- Testing execute_query_cmd ---")
    # sample_data is defined above:
    # sample_data = [
    #     {"id": 1, "name": "Alice", "age": 30, "city": "New York"},
    #     {"id": 2, "name": "Bob", "age": 24, "occupation": "Engineer"},
    #     {"id": 3, "name": "Charlie", "city": "London", "country": "UK"},
    #     {"id": 4, "name": "Diana"} 
    # ]

    test_queries = [
        ("SELECT name, age FROM sample_data WHERE age > 25", 
         # Expected: [{"name": "Alice", "age": 30}]
        ),
        ("SELECT * FROM sample_data WHERE city = 'New York'",
         # Expected: [{"id": 1, "name": "Alice", "age": 30, "city": "New York"}]
        ),
        ("SELECT name, city FROM sample_data ORDER BY name ASC",
         # Expected: [{"name": "Alice", "city": "New York"}, {"name": "Bob"}, {"name": "Charlie", "city": "London"}, {"name": "Diana"}] (Bob has no city)
        ),
        ("SELECT id, name, occupation FROM sample_data WHERE age <= 30 ORDER BY age DESC",
         # Expected: [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob", "occupation": "Engineer"}] (Alice has no occupation)
        ),
        ("SELECT name FROM sample_data WHERE country = 'UK'",
         # Expected: [{"name": "Charlie"}]
        ),
        ("SELECT * FROM sample_data ORDER BY id DESC",
         # Expected: All records, reverse order by id
        ),
        # Malformed queries for ValueError
        ("SELECT name age FROM sample_data", "Malformed SQL query"), # Missing comma
        ("SELECT name FROM sample_data WHERE age >", "Malformed WHERE clause"), # Incomplete WHERE
        ("SELECT name FROM sample_data ORDER BY name OTHER", "Malformed ORDER BY clause"), # Invalid ORDER BY
        ("SELECT name FROM sample_data WHERE non_existent_field = 10", "Field 'non_existent_field' not found"), # Field not found in WHERE (current behavior is to skip, not error)
                                                                                                                # Let's adjust test for current behavior: it should return empty if no record has the field or matches
        ("SELECT name FROM sample_data WHERE age = 'abc'", "Type mismatch"), # Type mismatch in WHERE
        ("SELECT name FROM sample_data WHERE city !< 'London'", "Unsupported operator"), # Invalid operator
    ]

    for i, query_test in enumerate(test_queries):
        query_str = query_test[0]
        expected_outcome_hint = query_test[1] if len(query_test) > 1 else "execute"
        
        print(f"\nTest {i+1}: Query: \"{query_str}\"")
        print(f"Expected outcome hint: {expected_outcome_hint}")
        try:
            result = execute_query_cmd(sample_data, query_str)
            print("Result:")
            if result:
                for row in result:
                    print(row)
            else:
                print("[] (Empty result set)")
        except ValueError as e:
            print(f"Caught ValueError: {e}")

    # Specific test for WHERE field not found (current behavior: condition is false, record skipped)
    print("\nTest: WHERE clause with a field not present in any record (e.g., 'salary > 50000')")
    query_non_existent_where_field = "SELECT name FROM sample_data WHERE salary > 50000"
    try:
        result = execute_query_cmd(sample_data, query_non_existent_where_field)
        print("Result:")
        if result:
            for row in result: print(row)
        else:
            print("[] (Empty result set - as expected, since 'salary' field leads to false condition)")
    except ValueError as e:
        print(f"Caught ValueError: {e}")

    # Test for SELECT non-existent field (current behavior: field is omitted from result)
    print("\nTest: SELECT a field not present in some records (e.g., 'occupation')")
    query_select_sometimes_missing_field = "SELECT name, occupation FROM sample_data WHERE age > 20 ORDER BY name ASC"
    # Expected: Alice (no occupation), Bob (Engineer), Charlie (no occupation)
    try:
        result = execute_query_cmd(sample_data, query_select_sometimes_missing_field)
        print("Result:")
        for row in result: print(row)
    except ValueError as e:
        print(f"Caught ValueError: {e}")
