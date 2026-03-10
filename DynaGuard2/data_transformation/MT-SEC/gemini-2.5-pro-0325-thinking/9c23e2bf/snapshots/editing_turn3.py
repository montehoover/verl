import re
import operator
from functools import partial # As requested

def _parse_where_condition(condition_str: str) -> tuple[str, callable, any]:
    """
    Parses a single condition string like "age > 30" or "name = 'John Doe'".
    Returns a tuple (field_name, operator_function, value).
    Raises ValueError for malformed condition strings.
    """
    # Regex to capture: field, operator, and value (either quoted string or number)
    match = re.match(r"^\s*(\w+)\s*(!=|>=|<=|=|>|<)\s*(?:'([^']*)'|(\d+(?:\.\d+)?))\s*$", condition_str.strip())
    
    if not match:
        raise ValueError(f"Malformed condition string: '{condition_str}'")

    field, op_str, string_val, num_val_str = match.groups()

    value: any
    if string_val is not None:
        value = string_val
    elif num_val_str is not None:
        if '.' in num_val_str:
            value = float(num_val_str)
        else:
            value = int(num_val_str)
    else:
        # This case should ideally not be reached if regex is robust
        raise ValueError(f"Could not parse value from condition: '{condition_str}'")

    op_map = {
        "=": operator.eq, "!=": operator.ne,
        ">": operator.gt, "<": operator.lt,
        ">=": operator.ge, "<=": operator.le,
    }
    
    # op_str is guaranteed by regex to be one of the keys if match is successful
    return field, op_map[op_str], value

def run_custom_query(dataset: list[dict], query: str) -> list[dict]:
    """
    Processes a dataset using a SQL-like query string.
    Supports SELECT, WHERE, and ORDER BY operations.

    Args:
        dataset: A list of dictionaries.
        query: A SQL-like query string (e.g., "SELECT name, age WHERE age > 30 ORDER BY name ASC").

    Returns:
        A new list of dictionaries representing the query result.

    Raises:
        ValueError: If the query is malformed or cannot be processed.
    """
    query_norm = query.strip()
    query_upper = query_norm.upper()

    # Find clause indices
    select_idx = query_upper.find("SELECT ")
    where_idx = query_upper.find(" WHERE ")
    orderby_idx = query_upper.find(" ORDER BY ")

    if select_idx == -1:
        raise ValueError("Query must contain a SELECT clause.")

    # Validate clause order
    if where_idx != -1 and where_idx < select_idx:
        raise ValueError("WHERE clause cannot precede SELECT clause.")
    if orderby_idx != -1:
        if orderby_idx < select_idx:
            raise ValueError("ORDER BY clause cannot precede SELECT clause.")
        if where_idx != -1 and orderby_idx < where_idx:
            raise ValueError("ORDER BY clause cannot precede WHERE clause.")

    # Parse SELECT clause
    select_end_idx = len(query_norm)
    if where_idx != -1: select_end_idx = min(select_end_idx, where_idx)
    if orderby_idx != -1: select_end_idx = min(select_end_idx, orderby_idx)
    
    select_fields_str = query_norm[select_idx + len("SELECT "):select_end_idx].strip()
    if not select_fields_str:
        raise ValueError("SELECT clause is empty.")
    
    select_all_fields = False
    selected_fields_list: list[str] = []
    if select_fields_str == "*":
        select_all_fields = True
    else:
        selected_fields_list = [f.strip() for f in select_fields_str.split(',')]
        if not all(f and re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", f) for f in selected_fields_list):
            raise ValueError("Malformed field names in SELECT clause. Fields must be valid identifiers.")

    # Parse WHERE clause
    parsed_conditions = []
    if where_idx != -1:
        where_start_idx = where_idx + len(" WHERE ")
        where_end_idx_actual = orderby_idx if orderby_idx != -1 else len(query_norm)
        where_clause_full_str = query_norm[where_start_idx:where_end_idx_actual].strip()
        
        if not where_clause_full_str:
            raise ValueError("WHERE clause is present but empty.")
        
        condition_strings = re.split(r"\s+AND\s+", where_clause_full_str, flags=re.IGNORECASE)
        
        for cond_str in condition_strings:
            cond_str_stripped = cond_str.strip()
            if not cond_str_stripped:
                raise ValueError("Empty condition found in WHERE clause (e.g., due to 'AND AND' or trailing 'AND').")
            try:
                field, op_func, val = _parse_where_condition(cond_str_stripped)
                parsed_conditions.append({'field': field, 'op': op_func, 'value': val})
            except ValueError as e:
                raise ValueError(f"Error parsing WHERE condition '{cond_str_stripped}': {e}")
        if not parsed_conditions and where_clause_full_str:
             raise ValueError(f"Could not parse any valid conditions from WHERE clause: '{where_clause_full_str}'")


    # Parse ORDER BY clause
    order_by_field = None
    order_by_desc = False
    if orderby_idx != -1:
        orderby_start_idx = orderby_idx + len(" ORDER BY ")
        orderby_clause_str = query_norm[orderby_start_idx:].strip()
        
        match = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+(ASC|DESC))?$", orderby_clause_str, re.IGNORECASE)
        if not match:
            raise ValueError(f"Malformed ORDER BY clause: '{orderby_clause_str}'. Expected 'field [ASC|DESC]'.")
        order_by_field = match.group(1)
        if match.group(2) and match.group(2).upper() == "DESC":
            order_by_desc = True

    # --- Execution ---
    current_data = list(dataset) # Work on a copy

    # 1. Apply WHERE conditions
    if parsed_conditions:
        filtered_data = []
        for record in current_data:
            all_conditions_met = True
            for p_cond in parsed_conditions:
                record_val = record.get(p_cond['field'])
                try:
                    # Ensure types are somewhat compatible for comparison or let operator handle it
                    if record_val is None and p_cond['value'] is not None : # Comparing None with a value
                         all_conditions_met = False # Or specific logic for NULL comparison if defined
                         break
                    if not p_cond['op'](record_val, p_cond['value']):
                        all_conditions_met = False
                        break
                except TypeError: # Handles comparing truly incompatible types (e.g., dict > int)
                    all_conditions_met = False 
                    break 
            if all_conditions_met:
                filtered_data.append(record)
        current_data = filtered_data

    # 2. Apply SELECT fields
    if not select_all_fields:
        projected_data = []
        for record in current_data:
            new_record = {}
            for field in selected_fields_list:
                if field in record:
                    new_record[field] = record[field]
                # SQL typically includes selected fields even if they are all NULL.
                # Here, if a field is not in the record, it's omitted from new_record.
                # To behave like SQL SELECT non_existent_field (column of NULLs):
                # else: new_record[field] = None 
                # Current behavior: only existing fields are projected.
            projected_data.append(new_record)
        current_data = projected_data
    
    # 3. Apply ORDER BY
    if order_by_field:
        try:
            # Sorting with a key that handles None values to group them
            # Python's default sort places None before other types.
            # (item.get(order_by_field) is None) makes Nones (True) sort after non-Nones (False) if not reversed.
            # To sort Nones first: use `(item.get(order_by_field) is None, item.get(order_by_field))`
            # and adjust reverse logic or tuple elements.
            # Simpler: rely on default None handling and catch TypeErrors for mixed types.
            current_data.sort(key=lambda item: item.get(order_by_field, None), reverse=order_by_desc)
        except TypeError:
            raise ValueError(f"Cannot sort by field '{order_by_field}': values are of mixed, non-comparable types or field is missing inconsistently.")
            
    return current_data
