import re
import operator
import logging # Added for logging
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

def _parse_query(query_str: str, schema_keys: set) -> dict:
    """Parses the SQL-like query string into components."""
    parsed_query = {
        "select_fields": None,
        "where_condition": None,
        "order_by_condition": None,
        "is_select_all": False
    }

    # --- Parse SELECT clause ---
    select_match = re.search(r"SELECT\s+(.+?)(?:\s+WHERE|\s+ORDER BY|$)", query_str, re.IGNORECASE)
    if not select_match:
        raise ValueError("Invalid query: SELECT clause missing or malformed.")
    
    select_fields_str = select_match.group(1).strip()
    parsed_query["is_select_all"] = (select_fields_str == "*")
    
    if not parsed_query["is_select_all"]:
        _selected_columns = [field.strip() for field in select_fields_str.split(',')]
        for col in _selected_columns:
            if col not in schema_keys:
                raise ValueError(f"Invalid query: Column '{col}' in SELECT not found in data schema.")
        parsed_query["select_fields"] = _selected_columns
    else:
        parsed_query["select_fields"] = list(schema_keys) # Tentative, might change if data is filtered

    # --- Parse WHERE clause (optional) ---
    where_match = re.search(r"WHERE\s+(.+?)(?:\s+ORDER BY|$)", query_str, re.IGNORECASE)
    if where_match:
        where_clause_str = where_match.group(1).strip()
        condition_match = re.match(r"(\w+)\s*([<>=!]+)\s*(.+)", where_clause_str, re.IGNORECASE)
        if not condition_match:
            raise ValueError(f"Invalid WHERE clause format: '{where_clause_str}'. Expected 'column operator value'.")

        where_field, where_op_str, where_val_str = condition_match.groups()
        
        if where_field not in schema_keys:
             raise ValueError(f"Invalid WHERE clause: Column '{where_field}' not found in data schema.")
        if where_op_str not in OPERATORS:
            raise ValueError(f"Invalid WHERE clause: Unsupported operator '{where_op_str}'.")
        
        parsed_query["where_condition"] = {
            "field": where_field,
            "operator_func": OPERATORS[where_op_str],
            "value": _parse_value(where_val_str)
        }

    # --- Parse ORDER BY clause (optional) ---
    orderby_match = re.search(r"ORDER BY\s+(.+)", query_str, re.IGNORECASE)
    if orderby_match:
        orderby_clause_str = orderby_match.group(1).strip()
        orderby_parts = orderby_clause_str.split()
        orderby_field = orderby_parts[0]
        
        if orderby_field not in schema_keys:
             raise ValueError(f"Invalid ORDER BY clause: Column '{orderby_field}' not found in data schema.")

        orderby_direction = "ASC"
        if len(orderby_parts) > 1:
            direction_str = orderby_parts[1].upper()
            if direction_str not in ["ASC", "DESC"]:
                raise ValueError(f"Invalid ORDER BY direction: '{orderby_parts[1]}'. Expected ASC or DESC.")
            orderby_direction = direction_str
        
        parsed_query["order_by_condition"] = {
            "field": orderby_field,
            "reverse": (orderby_direction == "DESC")
        }
    return parsed_query

def _apply_where_clause(data: list[dict], where_condition: dict | None) -> list[dict]:
    """Filters data based on the WHERE condition."""
    if not where_condition:
        return data

    filtered_data = []
    where_field = where_condition["field"]
    where_operator_func = where_condition["operator_func"]
    parsed_where_value = where_condition["value"]

    for row in data:
        if where_field not in row:
            continue

        row_value = row[where_field]
        try:
            if isinstance(parsed_where_value, str):
                row_value_cmp, value_to_compare = str(row_value), parsed_where_value
            elif isinstance(parsed_where_value, int):
                row_value_cmp, value_to_compare = int(row_value), parsed_where_value
            elif isinstance(parsed_where_value, float):
                row_value_cmp, value_to_compare = float(row_value), parsed_where_value
            else:
                row_value_cmp, value_to_compare = row_value, parsed_where_value
        except (ValueError, TypeError):
            continue

        if where_operator_func(row_value_cmp, value_to_compare):
            filtered_data.append(row)
    return filtered_data

def _apply_order_by_clause(data: list[dict], order_by_condition: dict | None) -> list[dict]:
    """Sorts data based on the ORDER BY condition."""
    if not order_by_condition or not data:
        return data

    orderby_field = order_by_condition["field"]
    reverse_sort = order_by_condition["reverse"]
    
    try:
        # Using row.get() for robustness if a field is unexpectedly missing after filtering
        # However, _parse_query already validates orderby_field against the initial schema.
        data.sort(key=lambda row: row.get(orderby_field), reverse=reverse_sort)
    except TypeError as e:
        raise ValueError(f"Cannot sort by column '{orderby_field}' due to incompatible data types: {e}")
    return data

def _apply_select_clause(data: list[dict], select_fields: list[str], is_select_all: bool) -> list[dict]:
    """Projects data to the selected fields."""
    if not data:
        return []

    # If SELECT *, determine columns from the first row of (potentially filtered) data
    actual_select_fields = select_fields
    if is_select_all:
        if not data: # Should not happen if called after checks, but defensive
            return []
        actual_select_fields = list(data[0].keys())


    final_results = []
    for row in data:
        result_row = {}
        for col in actual_select_fields:
            if col not in row:
                # This implies heterogeneous data or an issue with SELECT * logic if columns vary
                raise ValueError(f"Column '{col}' selected but not found in processed row: {row}.")
            result_row[col] = row[col]
        final_results.append(result_row)
    return final_results

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
    # Initialize logger
    logger = logging.getLogger(__name__)
    # Configure logger if it has no handlers (to avoid duplicate handlers on multiple calls)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler() # Logs to console
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.info(f"Executing query: '{query}'")

    if not data:
        logger.info("Query executed on empty dataset. Returning empty list.")
        return []

    query_str_stripped = query.strip()
    # Make a copy of the data to avoid modifying the original list
    processed_data = list(data) 
    
    try:
        # Assume a schema based on the first row for column validation.
        schema_keys = set(data[0].keys()) # Use set for efficient lookups

        # 1. Parse the query
        parsed_query_components = _parse_query(query_str_stripped, schema_keys)
        logger.info(f"Parsed query components: {parsed_query_components}")

        # 2. Apply WHERE clause
        processed_data_after_where = _apply_where_clause(processed_data, parsed_query_components["where_condition"])
        logger.info(f"Data count after WHERE: {len(processed_data_after_where)}")


        # 3. Apply ORDER BY clause
        processed_data_after_order = _apply_order_by_clause(processed_data_after_where, parsed_query_components["order_by_condition"])
        # Logging count after order by is same as after where, so not strictly necessary unless debugging sort itself

        # 4. Apply SELECT clause
        select_fields_to_use = parsed_query_components["select_fields"]
        if parsed_query_components["is_select_all"] and not processed_data_after_order:
            logger.info("SELECT * on empty dataset after filtering/sorting. Returning empty list.")
            return []
        
        final_results = _apply_select_clause(processed_data_after_order, select_fields_to_use, parsed_query_components["is_select_all"])
        
        logger.info(f"Query executed successfully. Result count: {len(final_results)}")
        # For very large results, logging the full result might be too verbose.
        # Consider logging a summary or a sample if results can be extensive.
        # For now, logging up to first 5 results for brevity.
        if final_results:
            logger.debug(f"Query results (sample): {final_results[:5]}") 
        else:
            logger.debug("Query returned no results.")
            
        return final_results
    except ValueError as e:
        logger.error(f"ValueError during query execution: {e}", exc_info=True)
        raise # Re-raise the exception after logging
    except Exception as e: # Catch any other unexpected error
        logger.error(f"Unexpected error during query execution: {e}", exc_info=True)
        # Depending on policy, you might re-raise a generic error or the original one
        raise ValueError(f"An unexpected error occurred: {e}")
