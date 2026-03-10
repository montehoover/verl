from collections import defaultdict
from typing import Callable, List, Dict
import re
import operator
from functools import partial # Included as requested

def select_fields(data_records: List[Dict], fields: List[str]) -> List[Dict]:
    """
    Extracts specific fields from a list of dictionaries.

    Args:
        data_records: A list of dictionaries, where each dictionary
                      represents a data record.
        fields: A list of strings, representing the field names to extract.

    Returns:
        A list of dictionaries, where each dictionary contains only the
        specified fields from the original record. If a requested field
        is not present in a record, it is omitted from the new dictionary
        for that record.
    """
    result = []
    for record in data_records:
        new_record = {}
        for field in fields:
            if field in record:
                new_record[field] = record[field]
        if new_record:  # Add record only if it's not empty after selection
            result.append(new_record)
        # If the requirement is to add an empty dict if no fields are found,
        # then the `if new_record:` check should be removed or adjusted.
        # Based on "containing only those fields", an empty dict is appropriate
        # if none of the selected fields are present.
        # Let's refine: if the intention is to always produce a dictionary for each input record,
        # even if it's empty because no selected fields were found, then the `if new_record`
        # is not quite right.
        # A more direct interpretation:
        # new_record = {field: record[field] for field in fields if field in record}
        # result.append(new_record)
        # This will add empty dicts if no fields match.
        # Let's go with the more direct interpretation.

    # Revised loop based on direct interpretation:
    processed_records = []
    for record in data_records:
        selected_record = {field: record[field] for field in fields if field in record}
        processed_records.append(selected_record)
    return processed_records

if __name__ == '__main__':
    # Example Usage:
    dataset = [
        {'id': 1, 'name': 'Alice', 'age': 30, 'city': 'New York'},
        {'id': 2, 'name': 'Bob', 'age': 24, 'occupation': 'Engineer'},
        {'id': 3, 'name': 'Charlie', 'city': 'London', 'age': 35},
        {'id': 4, 'name': 'David'} # Record with fewer fields
    ]

    fields_to_extract = ['name', 'age', 'city']

    selected_data = select_fields(dataset, fields_to_extract)
    for item in selected_data:
        print(item)

    # Expected output:
    # {'name': 'Alice', 'age': 30, 'city': 'New York'}
    # {'name': 'Bob', 'age': 24}
    # {'name': 'Charlie', 'city': 'London', 'age': 35}
    # {'name': 'David'}

    fields_to_extract_2 = ['occupation', 'id']
    selected_data_2 = select_fields(dataset, fields_to_extract_2)
    for item in selected_data_2:
        print(item)
    # Expected output:
    # {'id': 1}
    # {'id': 2, 'occupation': 'Engineer'}
    # {'id': 3}
    # {'id': 4}

    # Example with no matching fields for a record
    dataset_2 = [{'a': 1}, {'b': 2}]
    fields_3 = ['c']
    selected_data_3 = select_fields(dataset_2, fields_3)
    for item in selected_data_3:
        print(item)
    # Expected output:
    # {}
    # {}

def filter_data(data_records: List[Dict], condition: Callable[[Dict], bool]) -> List[Dict]:
    """
    Filters a list of records based on a given condition.

    Args:
        data_records: A list of dictionaries, where each dictionary
                      represents a data record.
        condition: A callable that takes a record (dictionary) as input
                   and returns True if the record satisfies the condition,
                   False otherwise.

    Returns:
        A list of dictionaries containing only the records that satisfy
        the condition.
    """
    return [record for record in data_records if condition(record)]

if __name__ == '__main__':
    # ... (previous example usage for select_fields remains the same)

    # Example Usage for filter_data:
    print("\n--- filter_data examples ---")
    dataset_for_filtering = [
        {'id': 1, 'name': 'Alice', 'age': 30, 'city': 'New York'},
        {'id': 2, 'name': 'Bob', 'age': 24, 'occupation': 'Engineer', 'city': 'San Francisco'},
        {'id': 3, 'name': 'Charlie', 'city': 'London', 'age': 35},
        {'id': 4, 'name': 'David', 'age': 24} 
    ]

    # Condition 1: Age greater than 25
    print("\nRecords where age > 25:")
    filtered_by_age = filter_data(dataset_for_filtering, lambda record: record.get('age', 0) > 25)
    for item in filtered_by_age:
        print(item)
    # Expected output:
    # {'id': 1, 'name': 'Alice', 'age': 30, 'city': 'New York'}
    # {'id': 3, 'name': 'Charlie', 'city': 'London', 'age': 35}

    # Condition 2: City is 'New York'
    print("\nRecords where city is 'New York':")
    filtered_by_city = filter_data(dataset_for_filtering, lambda record: record.get('city') == 'New York')
    for item in filtered_by_city:
        print(item)
    # Expected output:
    # {'id': 1, 'name': 'Alice', 'age': 30, 'city': 'New York'}

    # Condition 3: Has 'occupation' field and age is 24
    print("\nRecords with 'occupation' and age is 24:")
    filtered_complex = filter_data(dataset_for_filtering, 
                                   lambda record: 'occupation' in record and record.get('age') == 24)
    for item in filtered_complex:
        print(item)
    # Expected output:
    # {'id': 2, 'name': 'Bob', 'age': 24, 'occupation': 'Engineer', 'city': 'San Francisco'}

    # Condition 4: No records satisfy (e.g., age < 20)
    print("\nRecords where age < 20 (expecting none):")
    filtered_empty = filter_data(dataset_for_filtering, lambda record: record.get('age', 0) < 20)
    for item in filtered_empty: # Should not print anything
        print(item)
    if not filtered_empty:
        print("No records found (as expected).")

# --- execute_query_cmd and helpers ---

SUPPORTED_OPERATORS = {
    '=': operator.eq,
    '!=': operator.ne,
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
}

def _parse_value(value_str: str):
    """
    Parses a string value from a query into a Python type (str, int, float, bool).
    String literals must be quoted.
    """
    value_str = value_str.strip()
    if (value_str.startswith("'") and value_str.endswith("'")) or \
       (value_str.startswith('"') and value_str.endswith('"')):
        return value_str[1:-1]  # Return as string without quotes
    
    lower_val_str = value_str.lower()
    if lower_val_str == 'true':
        return True
    if lower_val_str == 'false':
        return False
    
    try:
        # Attempt to convert to int first, then float
        if '.' in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        # If not a number, boolean, or quoted string, it's an invalid value literal
        raise ValueError(f"Value '{value_str}' must be a number, a boolean (true/false), or a quoted string.")

def execute_query_cmd(dataset_list: List[Dict], sql_query: str) -> List[Dict]:
    """
    Executes an SQL-like query on a list of dictionaries.
    Supports SELECT, WHERE, and ORDER BY clauses.
    Example: "SELECT name, age WHERE age > 25 ORDER BY name ASC"
    """
    query_regex = re.compile(
        r"SELECT\s+(?P<fields>.*?)"
        r"(?:\s+WHERE\s+(?P<where_clause>.*?))?"
        r"(?:\s+ORDER\s+BY\s+(?P<orderby_clause>.*?))?$",
        re.IGNORECASE
    )

    match = query_regex.match(sql_query.strip())
    if not match:
        raise ValueError("Malformed SQL query. Expected format: 'SELECT fields [WHERE condition] [ORDER BY order_field [ASC|DESC]]'")

    parts = match.groupdict()
    select_fields_str = parts["fields"].strip()
    where_clause_str = parts["where_clause"].strip() if parts["where_clause"] else None
    orderby_clause_str = parts["orderby_clause"].strip() if parts["orderby_clause"] else None

    current_results = list(dataset_list) # Work on a copy

    # 1. Process WHERE clause
    if where_clause_str:
        # Simple WHERE: field operator value
        where_match = re.match(r"(\w+)\s*([<>=!]+)\s*(.*)", where_clause_str, re.IGNORECASE)
        if not where_match:
            raise ValueError(f"Malformed WHERE clause: '{where_clause_str}'. Expected 'field operator value'.")
        
        field, op_str, val_str = where_match.groups()
        field = field.strip()
        op_str = op_str.strip()
        # val_str is parsed by _parse_value later

        if op_str not in SUPPORTED_OPERATORS:
            raise ValueError(f"Unsupported operator in WHERE clause: '{op_str}'")
        
        operator_func = SUPPORTED_OPERATORS[op_str]
        
        try:
            value_to_compare = _parse_value(val_str)
        except ValueError as e:
            raise ValueError(f"Error parsing value in WHERE clause ('{val_str}'): {e}")

        def condition_func(record: Dict) -> bool:
            if field not in record:
                return False 
            
            record_value = record.get(field)

            if record_value is None: # SQL NULL comparison behavior
                return False
            
            # Attempt type-sensitive comparison
            # If query value is number, try to convert record's string value to number
            final_record_value = record_value
            
            if isinstance(value_to_compare, (int, float)) and isinstance(record_value, str):
                try:
                    fv = float(record_value)
                    final_record_value = int(fv) if fv.is_integer() else fv
                except ValueError:
                    return False # Record's string value is not a number, cannot compare with query number
            
            try:
                return operator_func(final_record_value, value_to_compare)
            except TypeError:
                # Catches comparison errors like int > str if conversion wasn't possible/done
                return False

        current_results = filter_data(current_results, condition_func)

    # 2. Process ORDER BY clause
    if orderby_clause_str:
        order_match = re.match(r"(\w+)(?:\s+(ASC|DESC))?", orderby_clause_str, re.IGNORECASE)
        if not order_match:
            raise ValueError(f"Malformed ORDER BY clause: '{orderby_clause_str}'. Expected 'field [ASC|DESC]'.")
        
        order_field, order_direction_str = order_match.groups()
        order_field = order_field.strip()
        
        reverse_order = bool(order_direction_str and order_direction_str.upper() == 'DESC')
            
        def sort_key_func(record: Dict):
            val = record.get(order_field)
            # Sort None values first (like NULLS FIRST), then sort by actual values
            # This ensures comparable types are passed to sort for non-None values.
            return (val is None, val) if reverse_order else (val is not None, val)


        try:
            # A more robust sort key that handles None and potentially mixed types if Python's default sort allows
            # Python 3 sort doesn't compare different types like int and str directly.
            # The key function should ensure elements are comparable or group them.
            # Using a tuple (is_none, value) helps group Nones.
            # (False, value) comes before (True, None) for ASC if Nones are last.
            # (True, value) comes before (False, None) for ASC if Nones are first.
            # Let's sort Nones first for ASC, last for DESC.
            # ASC: (val is None (True for None, False for value), val) -> Nones appear last.
            # To make Nones first for ASC: (val is not None (True for value, False for None), val)
            # Or, simpler: Python's default sort places None before other types if not reversed.
            # However, direct comparison of mixed types (e.g. int and str) in list will fail.
            # The key must return comparable types or handle this.
            # The (is_none, value) trick works if all non-None values are comparable.
            
            # Simpler key for sorting:
            # Treat None as a very small value for sorting purposes.
            # This requires all actual values to be comparable with each other.
            def sort_key_final(item):
                value = item.get(order_field)
                if value is None:
                    # For ASC, None should be "smallest". For DESC, "largest".
                    # This can be tricky. A common strategy is (is_present, value).
                    return (0, None) # Group Nones together, effectively first for most types
                return (1, value) # Group actual values after Nones

            current_results.sort(key=sort_key_final, reverse=reverse_order)

        except TypeError as e:
            raise ValueError(f"Error sorting by field '{order_field}': Incompatible data types for comparison in the sort column. {e}")

    # 3. Process SELECT clause
    if select_fields_str == "*":
        # No specific field selection needed, return all fields of the processed records
        pass 
    else:
        fields_to_select = [f.strip() for f in select_fields_str.split(',')]
        if not all(f for f in fields_to_select if f): # Check for empty field names like "f1,,f2"
            raise ValueError("Malformed SELECT clause: Contains empty or invalid field names.")
        current_results = select_fields(current_results, fields_to_select)
        
    return current_results

if __name__ == '__main__':
    # ... (previous example usage for select_fields and filter_data remains the same)

    # Example Usage for execute_query_cmd:
    print("\n--- execute_query_cmd examples ---")
    query_dataset = [
        {'id': 1, 'name': 'Alice', 'age': 30, 'city': 'New York', 'score': 85.5},
        {'id': 2, 'name': 'Bob', 'age': 24, 'occupation': 'Engineer', 'city': 'San Francisco', 'score': 92.0},
        {'id': 3, 'name': 'Charlie', 'city': 'London', 'age': 35, 'score': 78.3},
        {'id': 4, 'name': 'David', 'age': 24, 'score': None}, # Record with None score
        {'id': 5, 'name': 'Eve', 'age': 30, 'city': 'New York', 'occupation': 'Designer', 'score': 85.5}
    ]

    print("\nQuery 1: SELECT name, age WHERE age > 25 ORDER BY name ASC")
    results1 = execute_query_cmd(query_dataset, "SELECT name, age WHERE age > 25 ORDER BY name ASC")
    for item in results1: print(item)
    # Expected:
    # {'name': 'Alice', 'age': 30}
    # {'name': 'Charlie', 'age': 35}
    # {'name': 'Eve', 'age': 30}

    print("\nQuery 2: SELECT * WHERE city = 'New York' ORDER BY age DESC")
    results2 = execute_query_cmd(query_dataset, "SELECT * WHERE city = 'New York' ORDER BY age DESC")
    for item in results2: print(item)
    # Expected (order of fields in * might vary but content should be):
    # {'id': 1, 'name': 'Alice', 'age': 30, 'city': 'New York', 'score': 85.5}
    # {'id': 5, 'name': 'Eve', 'age': 30, 'city': 'New York', 'occupation': 'Designer', 'score': 85.5}

    print("\nQuery 3: SELECT occupation, name WHERE name = 'Bob'")
    results3 = execute_query_cmd(query_dataset, "SELECT occupation, name WHERE name = 'Bob'")
    for item in results3: print(item)
    # Expected:
    # {'occupation': 'Engineer', 'name': 'Bob'}
    
    print("\nQuery 4: SELECT name, city, age FROM dataset_for_filtering ORDER BY score DESC")
    results4 = execute_query_cmd(query_dataset, "SELECT name, city, age, score ORDER BY score DESC")
    for item in results4: print(item)
    # Expected (Nones sorted first by default with (0,None) key for DESC, or last if key is (value is None, value))
    # Let's adjust sort key for Nones last in DESC.
    # Sort key (1, value) for non-None, (0, None) for None. Reverse will put (1, value) first.
    # {'name': 'Bob', 'city': 'San Francisco', 'age': 24, 'score': 92.0}
    # {'name': 'Alice', 'city': 'New York', 'age': 30, 'score': 85.5}
    # {'name': 'Eve', 'city': 'New York', 'age': 30, 'score': 85.5}
    # {'name': 'Charlie', 'city': 'London', 'age': 35, 'score': 78.3}
    # {'name': 'David', 'city': None, 'age': 24, 'score': None} (city is None for David)

    print("\nQuery 5: SELECT name WHERE score >= 90.0")
    results5 = execute_query_cmd(query_dataset, "SELECT name WHERE score >= 90.0")
    for item in results5: print(item)
    # Expected:
    # {'name': 'Bob'}

    print("\nQuery 6: Malformed query (no fields)")
    try:
        execute_query_cmd(query_dataset, "SELECT WHERE age > 25")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\nQuery 7: Malformed WHERE clause")
    try:
        execute_query_cmd(query_dataset, "SELECT name WHERE age IS 25") # IS not supported operator
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print("\nQuery 8: SELECT name, age WHERE city = Boston (unquoted string literal)")
    try:
        execute_query_cmd(query_dataset, "SELECT name, age WHERE city = Boston")
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    print("\nQuery 9: SELECT * ORDER BY non_existent_field ASC")
    # This should sort, with non_existent_field being None for all, order might be stable.
    results9 = execute_query_cmd(query_dataset, "SELECT * ORDER BY non_existent_field ASC")
    # print("Results for Query 9 (order by non_existent_field):")
    # for item in results9: print(item) # Should be original order or stable sort of Nones
    print(f"Query 9 returned {len(results9)} items, order based on non_existent_field (all None).")


    print("\nQuery 10: SELECT name WHERE age = '24' (string literal for number comparison)")
    results10 = execute_query_cmd(query_dataset, "SELECT name WHERE age = '24'")
    for item in results10: print(item)
    # Expected (if '24' is successfully compared to int 24):
    # {'name': 'Bob'}
    # {'name': 'David'}
    # The current _parse_value turns '24' into string "24".
    # The condition_func tries to convert record_value (int 24) if value_to_compare is string.
    # No, it's: if value_to_compare (string "24") is string, and record_value (int 24) is number.
    # This path is not explicitly handled for conversion in condition_func.
    # operator_func(24, "24") for '==' is False in Python.
    # Let's refine condition_func for this.
    # If value_to_compare is string "24" (from _parse_value(" '24' ")), and record_value is int 24.
    # My condition_func:
    # `if isinstance(value_to_compare, (int, float)) and isinstance(record_value, str):` -> False
    # `return operator_func(final_record_value, value_to_compare)` -> `operator.eq(24, "24")` -> False.
    # This is correct behavior: 24 != "24". If query meant number, it should be `age = 24`.
    # So, Query 10 should return no results.
    if not results10:
        print("No records found for Query 10 (as expected, 24 != '24').")


    # Refined sort key for Query 4 expectation:
    # For ORDER BY score DESC, Nones should be last.
    # sort_key_final: (0, None) for None, (1, value) for value.
    # reverse=True: (1, value) comes before (0, None). So Nones are last. This is correct.
    # David has city=None, so it will appear as None in output.
    # Original dataset_for_filtering does not have 'score'. Using query_dataset.
    # David in query_dataset has no 'city' field. select_fields will omit it.
    # {'name': 'David', 'age': 24, 'score': None}
