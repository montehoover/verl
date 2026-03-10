import re
import operator
from functools import partial

def check_fields(dataset: list[dict], field_names: list[str], conditions: dict) -> bool:
    """
    Checks if any record in the dataset meets criteria based on field presence and conditions.
    A record is considered a match if:
    1. It contains at least one of the field names specified in `field_names`.
       (If `field_names` is empty, this criterion cannot be met by any record).
    2. AND it satisfies all key-value pair conditions specified in `conditions`.
       (If `conditions` is empty, this criterion is considered met for any record that passes criterion 1).

    Args:
        dataset: A list of dictionaries.
        field_names: A list of field names (strings). A record must contain at least one of these.
                     If empty, no record can meet the first criterion, and the function will return False.
        conditions: A dictionary specifying field-value pairs to match.
                    Example: {"age": 30, "city": "New York"}.
                    If empty, this part of the check is trivially satisfied.

    Returns:
        True if at least one record meets both criteria (field presence AND conditions), False otherwise.
    """
    for record in dataset:
        # Criterion 1: Record must contain at least one of the field_names
        has_matching_field = False
        # If field_names is empty, this loop won't run, has_matching_field remains False.
        for fn in field_names:
            if fn in record:
                has_matching_field = True
                break
        
        if not has_matching_field:
            continue  # This record doesn't meet the first criterion.

        # Criterion 2: Record must satisfy all conditions
        # If conditions is empty, this part is considered met.
        satisfies_all_conditions = True
        if conditions:  # Only iterate if conditions is not empty
            for cond_key, cond_value in conditions.items():
                if record.get(cond_key) != cond_value:
                    satisfies_all_conditions = False
                    break
        
        if satisfies_all_conditions:  # Implicitly, has_matching_field was also true to reach here
            return True  # Found a record meeting both criteria

    return False  # No record found that meets both criteria

def process_sql_request(dataset_records: list[dict], sql_statement: str) -> list[dict]:
    """
    Parses and executes a SQL-like query on a list of dictionary records.
    Supports SELECT, FROM, and WHERE clauses.
    - SELECT: specifies fields to return ('*' for all, or comma-separated field names).
    - FROM: specifies the dataset (name is ignored as dataset is passed directly).
    - WHERE: filters records based on conditions (e.g., 'age > 30 AND city = \'New York\'').
      - Supported operators: =, !=, >, <, >=, <=.
      - Values can be numbers or single-quoted strings.
      - Conditions are combined with AND.

    Args:
        dataset_records: A list of dictionaries representing the dataset.
        sql_statement: A string containing the SQL-like query.

    Returns:
        A list of dictionaries representing the query result.

    Raises:
        ValueError: If the SQL query is malformed or an error occurs during execution.
    """
    # Regex to capture SELECT, FROM, and optional WHERE. Case-insensitive.
    # Allows for optional semicolon and whitespace at start/end.
    query_pattern = re.compile(
        r"^\s*SELECT\s+(?P<fields>.*?)\s+FROM\s+\w+"
        r"(?:\s+WHERE\s+(?P<conditions>.*?))?\s*;?\s*$",
        re.IGNORECASE
    )
    match = query_pattern.fullmatch(sql_statement.strip())

    if not match:
        raise ValueError("Malformed SQL query: General structure error. Expected 'SELECT ... FROM ... [WHERE ...]'")

    query_parts = match.groupdict()
    select_fields_str = query_parts["fields"].strip()
    # conditions_str will be None if WHERE clause is not present
    conditions_str = query_parts.get("conditions")

    # Process SELECT fields
    selected_fields_list = []
    is_select_all = False
    if select_fields_str == "*":
        is_select_all = True
    else:
        selected_fields_list = [f.strip() for f in select_fields_str.split(',')]
        # Ensure no field name is empty after stripping and list is not empty
        if not selected_fields_list or not all(selected_fields_list):
            raise ValueError("Malformed SQL query: Invalid or empty field name(s) in SELECT clause.")

    # Parse WHERE conditions if present
    parsed_conditions_list = []
    if conditions_str:
        conditions_str = conditions_str.strip()
        if conditions_str: # Proceed only if conditions_str is not empty after stripping
            op_map = {
                "=": operator.eq,
                "!=": operator.ne,
                ">": operator.gt,
                "<": operator.lt,
                ">=": operator.ge,
                "<=": operator.le,
            }
            # Split conditions by AND, case-insensitive
            condition_parts = re.split(r"\s+AND\s+", conditions_str, flags=re.IGNORECASE)
            
            for cond_part_full in condition_parts:
                cond_part = cond_part_full.strip()
                if not cond_part: # Skip empty conditions (e.g., from "cond1 AND AND cond2")
                    continue

                # Regex for "field_name operator value"
                # Value can be a number (int/float, possibly negative) or a single-quoted string.
                cond_match = re.match(r"(\w+)\s*([=><!]+)\s*(?:'([^']*)'|(-?\d+(?:\.\d+)?|-?\d+))", cond_part)
                
                if not cond_match:
                    raise ValueError(f"Malformed WHERE condition: '{cond_part}'")
                
                field, op_str, str_val, num_val_str = cond_match.groups()

                if op_str not in op_map:
                    raise ValueError(f"Unsupported operator '{op_str}' in WHERE condition: '{cond_part}'")
                
                op_func = op_map[op_str]
                value_to_compare = None

                if str_val is not None:
                    value_to_compare = str_val
                elif num_val_str is not None:
                    try:
                        if '.' in num_val_str:
                            value_to_compare = float(num_val_str)
                        else:
                            value_to_compare = int(num_val_str)
                    except ValueError:
                        # This should ideally not happen if regex for number is robust
                        raise ValueError(f"Invalid numeric value '{num_val_str}' in WHERE condition: '{cond_part}'")
                else:
                    # Should not be reached if regex correctly captures a value type
                    raise ValueError(f"Could not parse value in WHERE condition: '{cond_part}'")

                parsed_conditions_list.append({"field": field, "op": op_func, "value": value_to_compare})
            
            if not parsed_conditions_list and conditions_str: # Check if WHERE clause had content but nothing was parsed
                raise ValueError("Malformed WHERE clause: Contains no valid conditions.")

    # Filter records based on parsed conditions
    filtered_records = []
    for record in dataset_records:
        match_all_conditions = True
        if parsed_conditions_list: # Only evaluate if there are conditions
            for p_cond in parsed_conditions_list:
                record_val = record.get(p_cond["field"])
                cond_val = p_cond["value"]

                if p_cond["field"] not in record: # Field in condition not in record
                    match_all_conditions = False
                    break
                
                try:
                    # Type compatibility for comparison
                    if isinstance(cond_val, str):
                        if not p_cond["op"](str(record_val), cond_val):
                            match_all_conditions = False
                            break
                    elif isinstance(cond_val, (int, float)):
                        comparable_record_val = record_val
                        if not isinstance(record_val, (int, float)):
                            try: # Attempt to convert record_val to the type of cond_val
                                comparable_record_val = type(cond_val)(record_val)
                            except (ValueError, TypeError): # Conversion failed
                                match_all_conditions = False
                                break
                        
                        if not p_cond["op"](comparable_record_val, cond_val):
                            match_all_conditions = False
                            break
                    else: # Should not happen with current value parsing
                        if not p_cond["op"](record_val, cond_val):
                            match_all_conditions = False
                            break
                except TypeError: # Catch unexpected comparison errors
                    match_all_conditions = False
                    break
            
        if match_all_conditions:
            filtered_records.append(record)
    
    # Construct final result based on SELECT fields
    final_result = []
    for record in filtered_records:
        if is_select_all:
            final_result.append(dict(record)) # Append a copy of the whole record
        else:
            new_record = {}
            for field_name in selected_fields_list:
                if field_name in record:
                    new_record[field_name] = record[field_name]
                else:
                    # SQL typically returns NULL for selected fields not present in a source row.
                    new_record[field_name] = None 
            final_result.append(new_record)
            
    return final_result
