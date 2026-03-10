import re
import operator
from functools import partial # As per problem setup, though not actively used in this version

# Global for operator mapping
_OPERATORS = {
    '=': operator.eq,
    '==': operator.eq, # Allow '==' for equality
    '!=': operator.ne,
    '<>': operator.ne, # Allow '<>' for inequality
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
}

def _parse_value(value_str):
    """Converts a string value from a query into its Python type."""
    value_str = value_str.strip()
    if (value_str.startswith("'") and value_str.endswith("'")) or \
       (value_str.startswith('"') and value_str.endswith('"')):
        return value_str[1:-1]  # String literal

    # Try to parse as number (float then int)
    if '.' in value_str:
        try:
            return float(value_str)
        except ValueError:
            # Not a float, could be malformed or an unquoted string.
            # Fall through to try int or raise specific error.
            pass
    try:
        return int(value_str)
    except ValueError:
        # If it's not a number and not quoted, it's an invalid format for a literal value.
        raise ValueError(f"Invalid value format: '{value_str}'. String values must be quoted. Numeric values must be valid numbers.")

def _parse_where_condition(condition_str):
    """Parses a single WHERE condition like 'age > 30' or 'name = "Alice"'."""
    # Regex to capture field, operator, and value. Value is 'everything else', parsed by _parse_value.
    match = re.match(r"^\s*(\w+)\s*([<>=!]+)\s*(.+)$", condition_str.strip())
    if not match:
        raise ValueError(f"Invalid WHERE condition format: '{condition_str}'")

    field, op_str, value_literal = match.groups()
    
    op_func = _OPERATORS.get(op_str)
    if not op_func:
        raise ValueError(f"Unsupported operator: '{op_str}' in condition '{condition_str}'")

    try:
        value = _parse_value(value_literal)
    except ValueError as e:
        # Re-raise with more context if _parse_value failed
        raise ValueError(f"Error parsing value in WHERE condition '{condition_str}': {e}")

    return {'field': field.strip(), 'op': op_func, 'value': value}


def run_sql_query(dataset, sql_query):
    """
    Processes a custom SQL-like query on data represented as a list of dictionaries.

    Args:
        dataset: A list of dictionaries where each dictionary represents a record.
        sql_query: A string containing the SQL-like query.

    Returns:
        A list of dictionaries representing the query results.

    Raises:
        ValueError: If the query is malformed, a field is not found,
                    or types are incompatible for an operation.
    """
    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a list.")
    if dataset and not all(isinstance(item, dict) for item in dataset):
        raise ValueError("All items in the dataset must be dictionaries.")

    query = sql_query.strip()
    if not query:
        raise ValueError("SQL query cannot be empty.")

    # Regex for SELECT, optional WHERE, optional ORDER BY
    # Groups: (1) select_cols, (2) where_clause_body, (3) order_by_clause_body
    query_match = re.match(
        r"SELECT\s+(.+?)"
        r"(?:\s+WHERE\s+(.+?))?"
        r"(?:\s+ORDER BY\s+(.+?))?$",
        query,
        re.IGNORECASE # For keywords SELECT, WHERE, ORDER BY
    )

    if not query_match:
        if not query.upper().startswith("SELECT "):
            raise ValueError("Query must start with SELECT.")
        raise ValueError("Invalid query structure. Expected format: SELECT cols [WHERE condition] [ORDER BY fields]")

    select_str, where_str, orderby_str = query_match.groups()

    # --- 1. Parse SELECT columns ---
    if not select_str: 
        raise ValueError("SELECT clause cannot be empty.") # Should be caught by regex `.+?`
    
    selected_columns_list = [col.strip() for col in select_str.split(',')]
    if not selected_columns_list or not all(col for col in selected_columns_list):
        raise ValueError("Invalid column specification in SELECT clause. Contains empty or invalid column names.")

    is_select_all = selected_columns_list == ['*']
    
    if not is_select_all and dataset: # Validate column names if dataset is not empty
        first_record_keys = dataset[0].keys()
        for col_name in selected_columns_list:
            if col_name not in first_record_keys:
                raise ValueError(f"Column '{col_name}' in SELECT clause not found in dataset records.")

    processed_data = list(dataset) # Work on a copy

    # --- 2. Apply WHERE clause ---
    if where_str:
        if " AND " in where_str.upper() or " OR " in where_str.upper():
            raise ValueError("Compound WHERE conditions (AND/OR) are not supported in this version.")
            
        try:
            condition = _parse_where_condition(where_str)
        except ValueError as e:
            raise ValueError(f"Error parsing WHERE clause ('{where_str}'): {e}")

        field_to_check = condition['field']
        op_func = condition['op']
        value_to_compare = condition['value']
        
        filtered_data = []
        for record_idx, record in enumerate(processed_data):
            if field_to_check not in record:
                raise ValueError(f"Field '{field_to_check}' in WHERE clause not found in record at index {record_idx}: {record}")

            record_value = record[field_to_check]
            
            try:
                if op_func(record_value, value_to_compare):
                    filtered_data.append(record)
            except TypeError:
                raise ValueError(
                    f"Type mismatch in WHERE condition for field '{field_to_check}'. "
                    f"Cannot compare record value '{record_value}' (type: {type(record_value).__name__}) "
                    f"with query value '{value_to_compare}' (type: {type(value_to_compare).__name__}) using operator '{op_func.__name__}'."
                )
        processed_data = filtered_data

    # --- 3. Apply ORDER BY clause ---
    if orderby_str and processed_data: # Only sort if there's data and an ORDER BY clause
        order_field_specs = []
        order_parts = [part.strip() for part in orderby_str.split(',')]

        for part in order_parts:
            if not part: continue # Skip empty parts like "name ASC, , age DESC"
            
            order_match = re.match(r"(\w+)\s*(ASC|DESC)?$", part, re.IGNORECASE) # For ASC/DESC
            if not order_match:
                raise ValueError(f"Invalid ORDER BY field format: '{part}'")
            
            col_name, direction_str = order_match.groups()
            
            if col_name not in processed_data[0].keys(): # Check against first record of (potentially filtered) data
                 raise ValueError(f"Column '{col_name}' in ORDER BY clause not found in dataset records.")
            
            is_descending = bool(direction_str and direction_str.upper() == "DESC")
            order_field_specs.append({'field': col_name, 'reverse': is_descending})

        for sort_spec in reversed(order_field_specs): # Apply sorts in reverse order of precedence for stability
            try:
                processed_data.sort(key=operator.itemgetter(sort_spec['field']), reverse=sort_spec['reverse'])
            except KeyError: 
                raise ValueError(f"Column '{sort_spec['field']}' for sorting not found consistently in records.")
            except TypeError:
                raise ValueError(f"Cannot sort by column '{sort_spec['field']}' due to incompatible (mixed) data types in that column.")

    # --- 4. Apply SELECT transformation (projection) ---
    if not is_select_all:
        final_result = []
        for record in processed_data:
            new_record = {}
            for col_name in selected_columns_list:
                if col_name not in record: # Should be caught by earlier validation if schema is consistent
                    raise ValueError(f"Column '{col_name}' selected but not found in processed record: {record}.")
                new_record[col_name] = record[col_name]
            final_result.append(new_record)
        processed_data = final_result
    
    return processed_data
