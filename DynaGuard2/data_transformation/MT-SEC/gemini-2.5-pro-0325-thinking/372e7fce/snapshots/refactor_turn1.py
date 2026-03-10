import re
import operator
from functools import partial # Included as per setup, though not actively used in this impl.

# Regex for parsing the overall SQL-like query structure.
# It captures 'select_columns', 'where_condition', and 'orderby_columns'.
# The FROM clause is matched but its content ignored.
_FULL_QUERY_REGEX = re.compile(
    r"SELECT\s+(?P<select_columns>.*?)"
    r"(?:\s+FROM\s+\w+\s*)?"  # Optional FROM clause (table name ignored)
    r"(?:\s+WHERE\s+(?P<where_condition>.*?)\s*)?"  # Optional WHERE clause
    r"(?:\s+ORDER BY\s+(?P<orderby_columns>.*?)\s*)?$",  # Optional ORDER BY clause
    re.IGNORECASE
)

def _parse_where_condition(condition_str: str):
    """Parses a single WHERE condition string."""
    # Using \S+ for column name to allow for non-alphanumeric characters like hyphens.
    match = re.match(r"(\S+)\s*([<>=!]+)\s*(.*)", condition_str.strip())
    if not match:
        raise ValueError(f"Invalid WHERE condition format: {condition_str}")
    
    col, op_str, val_str = match.groups()
    val_str = val_str.strip()

    if (val_str.startswith("'") and val_str.endswith("'")) or \
       (val_str.startswith('"') and val_str.endswith('"')):
        value = val_str[1:-1]
    else:
        try:
            value = int(val_str)
        except ValueError:
            try:
                value = float(val_str)
            except ValueError:
                raise ValueError(
                    f"Unsupported value format in WHERE clause: '{val_str}'. "
                    "Must be a number or a quoted string."
                )

    op_map = {
        "=": operator.eq, "!=": operator.ne,
        ">": operator.gt, "<": operator.lt,
        ">=": operator.ge, "<=": operator.le,
    }
    if op_str not in op_map:
        raise ValueError(f"Unsupported operator in WHERE clause: {op_str}")
    
    return col, op_map[op_str], value

def _apply_where(data: list[dict], where_clause_str: str) -> list[dict]:
    """Applies the WHERE clause to filter data."""
    if not data:
        return []
        
    try:
        column_name, op_func, condition_value = _parse_where_condition(where_clause_str)
    except ValueError as e:
        raise ValueError(f"Error parsing WHERE clause '{where_clause_str}': {e}")

    filtered_data = []
    for record in data:
        if column_name not in record:
            raise ValueError(f"Column '{column_name}' not found in dataset for WHERE clause.")
        
        record_value = record[column_name]
        
        try:
            # Attempt type coercion for comparison
            matches = False
            if isinstance(condition_value, (int, float)):
                # Condition is numeric, try to cast record value to this type
                record_value_coerced = type(condition_value)(record_value)
                if op_func(record_value_coerced, condition_value):
                    matches = True
            elif isinstance(condition_value, str):
                # Condition is string, cast record value to string
                if op_func(str(record_value), condition_value):
                    matches = True
            else: # Should not happen based on _parse_where_condition
                if op_func(record_value, condition_value): # Fallback to direct comparison
                    matches = True
            
            if matches:
                filtered_data.append(record)
        except (ValueError, TypeError):
            # Coercion failed (e.g., float('text')) or comparison of incompatible types.
            # Treat as a non-match.
            pass
                
    return filtered_data

def _parse_orderby_item(item_str: str):
    """Parses a single item in the ORDER BY clause (e.g., 'column_name DESC')."""
    parts = item_str.strip().split()
    if not parts:
        raise ValueError("Empty item in ORDER BY clause.")
    col = parts[0]
    order = "ASC"
    if len(parts) > 1:
        order_token = parts[1].upper()
        if order_token == "DESC":
            order = "DESC"
        elif order_token != "ASC":
            raise ValueError(f"Invalid ORDER BY direction: '{parts[1]}'. Must be ASC or DESC.")
    return col, order

def _apply_orderby(data: list[dict], orderby_clause_str: str) -> list[dict]:
    """Applies the ORDER BY clause to sort data."""
    if not data:
        return []

    order_by_items_str = [s.strip() for s in orderby_clause_str.split(',') if s.strip()]
    if not order_by_items_str: # Handles empty or whitespace-only orderby_clause_str
        return data

    parsed_orderby_items = []
    for item_str in order_by_items_str:
        try:
            col, order_dir = _parse_orderby_item(item_str)
            parsed_orderby_items.append({'col': col, 'reverse': order_dir == "DESC"})
        except ValueError as e:
            raise ValueError(f"Error parsing ORDER BY item '{item_str}': {e}")

    if not parsed_orderby_items:
        return data

    # Check column existence using the first record
    if data:
        first_record = data[0]
        for item in parsed_orderby_items:
            if item['col'] not in first_record:
                raise ValueError(f"Column '{item['col']}' not found for ORDER BY clause.")

    # Perform stable sort by applying criteria from right to left (least to most significant)
    current_sort_data = list(data)
    for item in reversed(parsed_orderby_items):
        col_name = item['col']
        is_reverse = item['reverse']
        
        def sort_key_func(record):
            val = record.get(col_name) # Use .get() for safety, though checked above
            if isinstance(val, (int, float)): return (0, val)
            if isinstance(val, str): return (1, val)
            if val is None: return (2, None) # Consistent handling for None
            return (3, val) # Other types

        try:
            current_sort_data.sort(key=sort_key_func, reverse=is_reverse)
        except TypeError as e:
            raise ValueError(
                f"Cannot sort by column '{col_name}' due to incompatible data types "
                f"not handled by the sort key: {e}"
            )
            
    return current_sort_data

def _apply_select(data: list[dict], select_clause_str: str) -> list[dict]:
    """Applies the SELECT clause to project columns."""
    select_clause_str = select_clause_str.strip()
    if not select_clause_str:
        raise ValueError("SELECT clause cannot be empty.")

    if select_clause_str == "*":
        return [dict(r) for r in data] # Return copies

    selected_columns = [col.strip() for col in select_clause_str.split(',') if col.strip()]
    if not selected_columns:
        raise ValueError(f"No valid columns in SELECT clause: '{select_clause_str}'")

    # Check column existence using the first record (if data exists)
    if data:
        first_record = data[0]
        for col in selected_columns:
            if col not in first_record:
                raise ValueError(f"Column '{col}' in SELECT clause not found in dataset.")
    elif not data: # No data, SELECT specific columns results in empty list
        return []


    projected_data = []
    for record in data:
        new_record = {}
        for col_name in selected_columns:
            # Assuming homogeneous data structure after the initial check.
            # If data can be heterogeneous, record.get(col_name, None) might be preferred.
            new_record[col_name] = record[col_name]
        projected_data.append(new_record)
        
    return projected_data

def execute_query_cmd(dataset_list: list[dict], sql_query: str) -> list[dict]:
    """
    Executes a basic SQL-like statement on a dataset (list of dictionaries).

    Handles SELECT, WHERE, and ORDER BY clauses.

    Args:
        dataset_list: A list of dictionaries, where each dictionary represents a data record.
        sql_query: A string containing the SQL-like statement.
                   Example: "SELECT column1, column2 WHERE column3 = 'value' ORDER BY column1 DESC"

    Returns:
        A list of dictionaries representing the results of the query operation.

    Raises:
        ValueError: If there is an issue with the query format or if the query
                    cannot be processed successfully (e.g., column not found,
                    invalid operator, type mismatch during comparisons).
    """
    if not isinstance(dataset_list, list):
        raise ValueError("Dataset must be a list of dictionaries.")
    if dataset_list and not all(isinstance(item, dict) for item in dataset_list):
        raise ValueError("All items in the dataset must be dictionaries.")
    if not isinstance(sql_query, str):
        raise ValueError("SQL query must be a string.")

    match = _FULL_QUERY_REGEX.match(sql_query.strip())
    if not match:
        raise ValueError(f"Invalid SQL query format: '{sql_query}'")

    query_parts = match.groupdict()
    select_columns_str = query_parts.get("select_columns")
    where_condition_str = query_parts.get("where_condition")
    orderby_columns_str = query_parts.get("orderby_columns")

    if not select_columns_str: # Should be caught by regex, but as a safeguard
        raise ValueError("SELECT clause is missing or empty.")

    current_data = list(dataset_list) # Work with a copy

    # 1. Apply WHERE clause (filters data)
    if where_condition_str:
        current_data = _apply_where(current_data, where_condition_str)

    # 2. Apply ORDER BY clause (sorts filtered data)
    if orderby_columns_str:
        current_data = _apply_orderby(current_data, orderby_columns_str)

    # 3. Apply SELECT clause (projects columns from sorted/filtered data)
    # This must be done last as WHERE and ORDER BY might need columns not in the final SELECT.
    selected_data = _apply_select(current_data, select_columns_str)
    
    return selected_data
