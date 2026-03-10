import re
import operator
from functools import partial

def handle_sql_query(records, sql_command):
    """
    Process a custom SQL-like query on data represented as a list of dictionaries.
    
    Args:
        records: A list of dictionaries where each dictionary represents a record
        sql_command: A string containing the SQL-like query to be executed
        
    Returns:
        A list containing dictionaries that represent the results of the executed query
        
    Raises:
        ValueError: If the query is not correctly formed or execution fails
    """
    if not sql_command:
        raise ValueError("SQL command cannot be empty")
    
    # Parse the SQL command
    sql_command = sql_command.strip()
    
    # Extract SELECT clause
    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_command, re.IGNORECASE)
    if not select_match:
        raise ValueError("Invalid SQL syntax: SELECT ... FROM clause not found")
    
    select_fields = [field.strip() for field in select_match.group(1).split(',')]
    
    # Extract WHERE clause (optional)
    where_match = re.search(r'WHERE\s+(.*?)(?:\s+ORDER\s+BY|$)', sql_command, re.IGNORECASE)
    where_condition = where_match.group(1).strip() if where_match else None
    
    # Extract ORDER BY clause (optional)
    order_match = re.search(r'ORDER\s+BY\s+(.*?)$', sql_command, re.IGNORECASE)
    order_clause = order_match.group(1).strip() if order_match else None
    
    # Process records
    result = records.copy()
    
    # Apply WHERE clause
    if where_condition:
        result = apply_where_clause(result, where_condition)
    
    # Apply ORDER BY clause
    if order_clause:
        result = apply_order_by_clause(result, order_clause)
    
    # Apply SELECT clause
    result = apply_select_clause(result, select_fields)
    
    return result


def apply_where_clause(records, condition):
    """Apply WHERE clause filtering to records."""
    # Parse condition (support simple conditions like "field > value")
    # This regex matches: field operator value
    match = re.match(r'(\w+)\s*(=|!=|<>|<=|>=|<|>)\s*(.+)', condition)
    if not match:
        raise ValueError(f"Invalid WHERE condition: {condition}")
    
    field, op, value = match.groups()
    value = value.strip()
    
    # Convert operator to function
    op_map = {
        '=': operator.eq,
        '!=': operator.ne,
        '<>': operator.ne,
        '<': operator.lt,
        '<=': operator.le,
        '>': operator.gt,
        '>=': operator.ge
    }
    
    if op not in op_map:
        raise ValueError(f"Unsupported operator: {op}")
    
    op_func = op_map[op]
    
    # Filter records
    filtered = []
    for record in records:
        if field not in record:
            continue
            
        record_value = record[field]
        
        # Try to convert value to match the type of record_value
        try:
            if isinstance(record_value, int):
                compare_value = int(value)
            elif isinstance(record_value, float):
                compare_value = float(value)
            else:
                # Remove quotes if present for string values
                compare_value = value.strip('"\'')
        except ValueError:
            # If conversion fails, use as string
            compare_value = value.strip('"\'')
        
        if op_func(record_value, compare_value):
            filtered.append(record)
    
    return filtered


def apply_order_by_clause(records, order_clause):
    """Apply ORDER BY clause to sort records."""
    # Parse order clause (support "field" or "field ASC" or "field DESC")
    parts = order_clause.split()
    if not parts:
        raise ValueError("Invalid ORDER BY clause")
    
    field = parts[0]
    descending = False
    
    if len(parts) > 1:
        if parts[1].upper() == 'DESC':
            descending = True
        elif parts[1].upper() != 'ASC':
            raise ValueError(f"Invalid ORDER BY direction: {parts[1]}")
    
    # Sort records
    try:
        return sorted(records, key=lambda x: x.get(field, None), reverse=descending)
    except Exception as e:
        raise ValueError(f"Error sorting by {field}: {str(e)}")


def apply_select_clause(records, fields):
    """Apply SELECT clause to project specific fields."""
    if not fields:
        raise ValueError("No fields specified in SELECT clause")
    
    # Handle SELECT *
    if len(fields) == 1 and fields[0] == '*':
        return records
    
    # Project specific fields
    result = []
    for record in records:
        projected = {}
        for field in fields:
            if field in record:
                projected[field] = record[field]
        if projected:  # Only add if at least one field was found
            result.append(projected)
    
    return result
