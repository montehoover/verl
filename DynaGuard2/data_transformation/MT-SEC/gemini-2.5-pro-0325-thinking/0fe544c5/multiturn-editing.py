import re
import operator
from functools import partial # Though not used in this version, imported as requested

# Operator mapping
OPS = {
    '=': operator.eq,
    '!=': operator.ne,
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
}

def _parse_value(value_str):
    """Converts a SQL value string to a Python type."""
    value_str = value_str.strip()
    if (value_str.startswith("'") and value_str.endswith("'")) or \
       (value_str.startswith('"') and value_str.endswith('"')):
        return value_str[1:-1]  # String literal
    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False
    try:
        return int(value_str)  # Integer
    except ValueError:
        try:
            return float(value_str)  # Float
        except ValueError:
            raise ValueError(f"Unknown value type for: {value_str}")

def _get_operator_fn(op_str):
    """Gets the operator function from the OPS map."""
    if op_str not in OPS:
        raise ValueError(f"Unsupported operator: {op_str}")
    return OPS[op_str]

def handle_sql_query(records, sql_command):
    """
    Parses and executes a SQL-like query on a list of dictionaries.

    Args:
        records: A list of dictionaries representing the dataset.
        sql_command: A string with a SQL-like query.
                     Example: "SELECT name, age FROM data WHERE age > 30 AND city = 'New York'"
                              "SELECT * FROM data WHERE occupation = 'Engineer'"

    Returns:
        A list of dictionaries representing the query result.

    Raises:
        ValueError: If the query is malformed, a condition references a non-existent field,
                    or an operator is unsupported.
    """
    sql_command = sql_command.strip()
    # Regex to parse SELECT <fields> FROM <table> [WHERE <conditions>]
    # We don't actually use the table name from the query, as `records` is our table.
    match = re.fullmatch(
        r"SELECT\s+(.+?)\s+FROM\s+\w+(?:\s+WHERE\s+(.+))?",
        sql_command,
        re.IGNORECASE
    )

    if not match:
        raise ValueError(f"Malformed SQL query: {sql_command}")

    select_fields_str, where_clause_str = match.groups()

    # Parse selected fields
    if select_fields_str.strip() == '*':
        selected_fields = None  # Indicates all fields
    else:
        selected_fields = [f.strip() for f in select_fields_str.split(',')]
        if not all(selected_fields): # Check for empty field names from "col1, , col2"
             raise ValueError("Empty field name in SELECT clause.")


    parsed_conditions = []
    if where_clause_str:
        # Split conditions by AND (more complex logic for OR/parentheses would be needed)
        conditions_list = re.split(r'\s+AND\s+', where_clause_str, flags=re.IGNORECASE)
        for cond_str in conditions_list:
            cond_match = re.match(r"(\w+)\s*([<>=!]+)\s*(.+)", cond_str.strip())
            if not cond_match:
                raise ValueError(f"Malformed condition: {cond_str}")
            
            field, op_str, val_str = cond_match.groups()
            op_fn = _get_operator_fn(op_str)
            value = _parse_value(val_str)
            parsed_conditions.append({'field': field, 'operator': op_fn, 'value': value})

    result_records = []
    for record in records:
        match_all_conditions = True
        if parsed_conditions:
            for cond in parsed_conditions:
                field_name = cond['field']
                if field_name not in record:
                    # As per requirement: "raise a ValueError if a condition references a non-existent field."
                    # This applies if the field for condition is missing.
                    raise ValueError(
                        f"Condition field '{field_name}' not found in record: {record}"
                    )
                
                record_value = record[field_name]
                # Type check: if record_value and cond['value'] are of incompatible types for comparison
                # For example, trying to compare a string with an int using > or <
                if isinstance(cond['value'], (int, float)) and not isinstance(record_value, (int, float)):
                    # Allow string comparison for equality/inequality
                    if cond['operator'] not in (operator.eq, operator.ne):
                         raise ValueError(
                            f"Type mismatch for field '{field_name}': cannot compare {type(record_value).__name__} with {type(cond['value']).__name__} using operator {cond['operator'].__name__}"
                        )
                elif isinstance(cond['value'], str) and not isinstance(record_value, str):
                     if cond['operator'] not in (operator.eq, operator.ne):
                        raise ValueError(
                            f"Type mismatch for field '{field_name}': cannot compare {type(record_value).__name__} with {type(cond['value']).__name__} using operator {cond['operator'].__name__}"
                        )


                if not cond['operator'](record_value, cond['value']):
                    match_all_conditions = False
                    break
        
        if match_all_conditions:
            if selected_fields is None:  # SELECT *
                result_records.append(dict(record)) # Append a copy
            else:
                new_item = {}
                for sf in selected_fields:
                    if sf in record:
                        new_item[sf] = record[sf]
                    else:
                        # SQL standard behavior: if a selected field doesn't exist, it's an error or returns NULL.
                        # Here, we can choose to raise error or add it as None or skip.
                        # For simplicity, let's raise an error if a specifically selected field is missing.
                        # Or, more leniently, just don't include it / include as None.
                        # Let's be strict for now, as with condition fields.
                        raise ValueError(f"Selected field '{sf}' not found in record: {record}")
                result_records.append(new_item)

    return result_records


def extract_fields(dataset, fields, conditions=None):
    """
    Extracts specified fields from a list of dictionaries and filters
    records based on given conditions.

    Args:
        dataset: A list of dictionaries.
        fields: A list of field names to extract.
        conditions: A dictionary of conditions to filter by.
                    Example: {'age': 30, 'city': 'New York'}
                    If None, no filtering is applied.

    Returns:
        A new list of dictionaries, where each dictionary contains only the
        specified fields and matches the conditions.

    Raises:
        ValueError: If a condition references a non-existent field in an item.
    """
    new_dataset = []
    for item in dataset:
        match = True
        if conditions:
            for cond_field, cond_value in conditions.items():
                if cond_field not in item:
                    raise ValueError(
                        f"Condition field '{cond_field}' not found in item: {item}"
                    )
                if item[cond_field] != cond_value:
                    match = False
                    break
        
        if match:
            new_item = {}
            for field in fields:
                if field in item:
                    new_item[field] = item[field]
            if new_item or not fields: # Add if new_item is not empty or if no specific fields were requested (meaning, return full matched item)
                new_dataset.append(new_item if fields else item) # if fields is empty, append the whole item
    return new_dataset

if __name__ == '__main__':
    # Example Usage
    data = [
        {'name': 'Alice', 'age': 30, 'city': 'New York'},
        {'name': 'Bob', 'age': 24, 'city': 'San Francisco'},
        {'name': 'Charlie', 'age': 35, 'city': 'London', 'occupation': 'Engineer'}
    ]

    fields_to_extract = ['name', 'city']
    extracted_data = extract_fields(data, fields_to_extract)
    print("Original dataset:")
    for row in data:
        print(row)
    print("\nExtracted dataset (name and city, no filter):")
    for row in extracted_data:
        print(row)

    fields_to_extract_with_occupation = ['name', 'occupation']
    extracted_data_occupation = extract_fields(data, fields_to_extract_with_occupation)
    print("\nExtracted dataset (name and occupation, no filter):")
    for row in extracted_data_occupation:
        print(row)

    # Example with filtering
    conditions_age_30 = {'age': 30}
    extracted_filtered_data_age = extract_fields(data, ['name', 'age', 'city'], conditions_age_30)
    print("\nExtracted dataset (name, age, city) for age = 30:")
    for row in extracted_filtered_data_age:
        print(row)

    conditions_london = {'city': 'London'}
    extracted_filtered_data_london = extract_fields(data, ['name', 'occupation'], conditions_london)
    print("\nExtracted dataset (name, occupation) for city = London:")
    for row in extracted_filtered_data_london:
        print(row)
    
    conditions_bob_sf = {'name': 'Bob', 'city': 'San Francisco'}
    extracted_filtered_data_bob = extract_fields(data, ['name', 'age', 'city'], conditions_bob_sf)
    print("\nExtracted dataset (name, age, city) for name = Bob and city = San Francisco:")
    for row in extracted_filtered_data_bob:
        print(row)

    # Example of filtering with a non-existent field in one of the items for condition
    print("\nAttempting to filter with a condition on 'country' (which doesn't exist for all):")
    try:
        conditions_country = {'country': 'USA'}
        extract_fields(data, ['name'], conditions_country)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Example of filtering with a non-existent field in one of the items for condition,
    # but the item that would cause an error is already filtered out by a previous valid condition.
    # This case should not raise an error if the item causing issues is filtered out before the problematic check.
    # However, the current implementation checks all conditions for an item before deciding to filter it out.
    # Let's test with a condition that will filter out Alice, then check for a field only Charlie has.
    print("\nAttempting to filter where 'age' is 24, then check 'occupation' (Alice doesn't have occupation):")
    try:
        # This will process Bob (age 24), then Charlie.
        # Charlie has 'occupation', Bob does not.
        # If conditions were {'age': 24, 'occupation': 'Engineer'}, it would fail on Bob.
        # If conditions were {'occupation': 'Engineer', 'age': 24}, it would fail on Alice.
        # The order of items in `data` and keys in `conditions` can matter for when the error is raised.
        
        # Let's try to filter for Bob, who doesn't have 'occupation'
        conditions_bob_occupation = {'name': 'Bob', 'occupation': 'Student'} # Bob has no 'occupation'
        extract_fields(data, ['name', 'age'], conditions_bob_occupation)
    except ValueError as e:
        print(f"Caught expected error (Bob has no occupation): {e}")

    print("\nAttempting to filter for Charlie using occupation, then a non-existent field for Charlie:")
    try:
        conditions_charlie_bad_field = {'occupation': 'Engineer', 'non_existent_field': 'value'}
        extract_fields(data, ['name', 'age'], conditions_charlie_bad_field)
    except ValueError as e:
        print(f"Caught expected error (Charlie has no non_existent_field): {e}")

    # Example: extract all fields for items matching condition
    print("\nExtracted all fields for city = New York:")
    extracted_all_fields_ny = extract_fields(data, [], {'city': 'New York'})
    for row in extracted_all_fields_ny:
        print(row)

    # --- Examples for handle_sql_query ---
    print("\n--- handle_sql_query examples ---")
    print("\nQuery: SELECT name, age FROM data WHERE age > 25")
    try:
        query_result_1 = handle_sql_query(data, "SELECT name, age FROM data WHERE age > 25")
        for row in query_result_1:
            print(row)
    except ValueError as e:
        print(f"Error: {e}")

    print("\nQuery: SELECT * FROM data WHERE city = 'London'")
    try:
        query_result_2 = handle_sql_query(data, "SELECT * FROM data WHERE city = 'London'")
        for row in query_result_2:
            print(row)
    except ValueError as e:
        print(f"Error: {e}")

    print("\nQuery: SELECT name, occupation FROM data WHERE city = 'New York' AND age = 30")
    try:
        # Alice is 30 and in New York, but 'occupation' is not a field for Alice.
        # The current implementation will raise error if selected field is not present.
        # Let's change query to select fields that Alice has.
        query_result_3 = handle_sql_query(data, "SELECT name, city, age FROM data WHERE city = 'New York' AND age = 30")
        for row in query_result_3:
            print(row)
    except ValueError as e:
        print(f"Error: {e}")
        
    print("\nQuery: SELECT name, occupation FROM data WHERE name = 'Charlie'")
    try:
        query_result_charlie_occupation = handle_sql_query(data, "SELECT name, occupation FROM data WHERE name = 'Charlie'")
        for row in query_result_charlie_occupation:
            print(row)
    except ValueError as e:
        print(f"Error: {e}")

    print("\nQuery with non-existent field in WHERE: SELECT name FROM data WHERE country = 'USA'")
    try:
        handle_sql_query(data, "SELECT name FROM data WHERE country = 'USA'")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\nQuery with non-existent selected field: SELECT name, non_existent_field FROM data WHERE age > 30")
    try:
        # This will first filter (Alice, Charlie), then try to select 'non_existent_field'.
        handle_sql_query(data, "SELECT name, non_existent_field FROM data WHERE age > 30")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\nMalformed query: SELECT name age FROM data")
    try:
        handle_sql_query(data, "SELECT name age FROM data")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\nQuery with unsupported operator: SELECT name FROM data WHERE age !! 30")
    try:
        handle_sql_query(data, "SELECT name FROM data WHERE age !! 30")
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    print("\nQuery with type mismatch for comparison: SELECT name FROM data WHERE name > 20")
    try:
        handle_sql_query(data, "SELECT name FROM data WHERE name > 20") # name is str, 20 is int
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\nQuery: SELECT * FROM data WHERE name = 'Bob' AND city = 'San Francisco'")
    try:
        query_result_bob = handle_sql_query(data, "SELECT * FROM data WHERE name = 'Bob' AND city = 'San Francisco'")
        for row in query_result_bob:
            print(row)
    except ValueError as e:
        print(f"Error: {e}")
