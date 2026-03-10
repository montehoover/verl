import re
import operator
from functools import partial # Included as per setup, though not actively used in this implementation

# Operator mapping for WHERE clause
OPERATORS = {
    '=': operator.eq,
    '!=': operator.ne,
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
}

def _parse_value(value_str: str):
    """Attempt to parse value string into int, float, or keep as string."""
    value_str = value_str.strip()
    if (value_str.startswith("'") and value_str.endswith("'")) or \
       (value_str.startswith('"') and value_str.endswith('"')):
        return value_str[1:-1]  # String literal

    try:
        return int(value_str)
    except ValueError:
        try:
            return float(value_str)
        except ValueError:
            # If it's not a quoted string, int, or float, it might be an unquoted string literal
            # or a field name on the right side of an operator (not supported here).
            # For simplicity, treat as a literal string if not parsable as number.
            return value_str

def execute_custom_query(data: list[dict], query: str) -> list[dict]:
    """
    Executes a custom SQL-like query on a list of dictionaries.

    Args:
        data: List of dictionaries representing the dataset.
        query: str, a SQL-like query string.

    Returns:
        List of dictionaries representing the query results.

    Raises:
        ValueError: Raised when the query is invalid or cannot be executed.
    """
    if not data:
        return []

    query = query.strip()
    # Make a copy of the data to avoid modifying the original list
    processed_data = list(data) 
    
    # Assume a schema based on the first row for column validation.
    # This is a simplification; real databases have explicit schemas.
    schema_keys = data[0].keys()

    # --- Parse SELECT clause ---
    # Regex captures: 1=fields
    select_match = re.search(r"SELECT\s+(.+?)(?:\s+WHERE|\s+ORDER BY|$)", query, re.IGNORECASE)
    if not select_match:
        raise ValueError("Invalid query: SELECT clause missing or malformed.")
    
    select_fields_str = select_match.group(1).strip()
    is_select_all = (select_fields_str == "*")
    
    _selected_columns = []
    if not is_select_all:
        _selected_columns = [field.strip() for field in select_fields_str.split(',')]
        for col in _selected_columns:
            if col not in schema_keys:
                raise ValueError(f"Invalid query: Column '{col}' in SELECT not found in data schema (based on first item).")
    # If is_select_all, _selected_columns will be determined after filtering.

    # --- Parse WHERE clause (optional) ---
    # Regex captures: 1=condition string
    where_match = re.search(r"WHERE\s+(.+?)(?:\s+ORDER BY|$)", query, re.IGNORECASE)
    if where_match:
        where_clause_str = where_match.group(1).strip()
        # Regex for "field op value": 1=field, 2=operator, 3=value
        condition_match = re.match(r"(\w+)\s*([<>=!]+)\s*(.+)", where_clause_str, re.IGNORECASE)
        if not condition_match:
            raise ValueError(f"Invalid WHERE clause format: '{where_clause_str}'. Expected 'column operator value'.")

        where_field, where_op_str, where_val_str = condition_match.groups()
        
        if where_field not in schema_keys:
             raise ValueError(f"Invalid WHERE clause: Column '{where_field}' not found in data schema.")

        if where_op_str not in OPERATORS:
            raise ValueError(f"Invalid WHERE clause: Unsupported operator '{where_op_str}'.")
        
        where_operator_func = OPERATORS[where_op_str]
        parsed_where_value = _parse_value(where_val_str)

        # Filter data
        filtered_data = []
        for row in processed_data:
            if where_field not in row: # Should ideally not happen if schema is consistent
                # Skip row if essential field for WHERE is missing
                continue

            row_value = row[where_field]
            
            try:
                # Attempt type coercion for comparison
                if isinstance(parsed_where_value, str):
                    row_value_cmp = str(row_value)
                    value_to_compare = parsed_where_value
                elif isinstance(parsed_where_value, int):
                    row_value_cmp = int(row_value)
                    value_to_compare = parsed_where_value
                elif isinstance(parsed_where_value, float):
                    row_value_cmp = float(row_value)
                    value_to_compare = parsed_where_value
                else: # Should not be reached if _parse_value is correct
                    row_value_cmp = row_value 
                    value_to_compare = parsed_where_value
            except (ValueError, TypeError):
                # If row_value cannot be converted to match parsed_where_value's type,
                # this row cannot satisfy the condition. Skip it.
                continue 

            if where_operator_func(row_value_cmp, value_to_compare):
                filtered_data.append(row)
        processed_data = filtered_data

    # --- Parse ORDER BY clause (optional) ---
    # Regex captures: 1=full order by string (e.g. "age DESC")
    orderby_match = re.search(r"ORDER BY\s+(.+)", query, re.IGNORECASE)
    if orderby_match and processed_data: # Only sort if there's data to sort
        orderby_clause_str = orderby_match.group(1).strip()
        orderby_parts = orderby_clause_str.split()
        orderby_field = orderby_parts[0]
        
        if orderby_field not in schema_keys: # Validate against original schema
             raise ValueError(f"Invalid ORDER BY clause: Column '{orderby_field}' not found in data schema.")

        orderby_direction = "ASC"
        if len(orderby_parts) > 1:
            direction_str = orderby_parts[1].upper()
            if direction_str in ["ASC", "DESC"]:
                orderby_direction = direction_str
            else:
                raise ValueError(f"Invalid ORDER BY direction: '{orderby_parts[1]}'. Expected ASC or DESC.")
        
        reverse_sort = (orderby_direction == "DESC")
        
        try:
            # Ensure the sorting key exists in all rows being sorted
            # Using row.get(orderby_field) can make it more robust to heterogeneous data
            # by treating missing keys as None (which might then require careful handling in sort).
            # However, for strict SQL-like behavior, missing sort key is an issue.
            # We check against schema_keys, but if data is truly heterogeneous,
            # a row in processed_data might lack orderby_field.
            processed_data.sort(key=lambda row: row.get(orderby_field), reverse=reverse_sort)

        except TypeError as e:
            # This happens if trying to sort items of mixed non-comparable types (e.g., int and string)
            raise ValueError(f"Cannot sort by column '{orderby_field}' due to incompatible data types in rows: {e}")

    # --- Apply SELECT projection ---
    if not processed_data: # If no data after filtering/sorting
        return []

    # Determine actual columns for SELECT * based on the (potentially filtered) data
    if is_select_all:
        # Use keys from the first item of the processed_data. Assumes homogeneity.
        _selected_columns = list(processed_data[0].keys()) 
    
    final_results = []
    for row in processed_data:
        result_row = {}
        for col in _selected_columns:
            if col not in row:
                # This implies heterogeneous data where a selected column is missing from a row.
                # SQL might return NULL here. For simplicity, we raise an error.
                raise ValueError(f"Column '{col}' selected but not found in processed row: {row}. Data may be heterogeneous.")
            result_row[col] = row[col]
        final_results.append(result_row)
            
    return final_results
