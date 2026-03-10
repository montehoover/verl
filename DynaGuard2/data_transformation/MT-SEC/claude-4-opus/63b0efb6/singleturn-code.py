import re
import operator
from functools import partial

def run_sql_query(records, command):
    """
    Execute a basic SQL-like statement on a dataset given as a list of dictionaries.
    
    Args:
        records: A list where each item is a dictionary representing data records
        command: A string containing the SQL-like statement for execution
        
    Returns:
        A list of dictionaries, which are the results of the query operation
        
    Raises:
        ValueError: If there is an issue with the query format or when the query can't be processed successfully
    """
    # Parse the SQL-like command
    command = command.strip()
    
    # Extract SELECT clause
    select_match = re.match(r'SELECT\s+(.+?)\s+FROM\s+\w+(?:\s+WHERE\s+(.+?))?(?:\s+ORDER\s+BY\s+(.+?))?$', command, re.IGNORECASE)
    
    if not select_match:
        raise ValueError("Invalid query format. Expected: SELECT ... FROM ... [WHERE ...] [ORDER BY ...]")
    
    select_clause = select_match.group(1)
    where_clause = select_match.group(2)
    order_by_clause = select_match.group(3)
    
    # Parse SELECT fields
    if select_clause.strip() == '*':
        select_fields = None  # Select all fields
    else:
        select_fields = [field.strip() for field in select_clause.split(',')]
    
    # Start with all records
    result = records.copy()
    
    # Apply WHERE clause if present
    if where_clause:
        result = apply_where_clause(result, where_clause)
    
    # Apply ORDER BY clause if present
    if order_by_clause:
        result = apply_order_by_clause(result, order_by_clause)
    
    # Apply SELECT clause
    if select_fields:
        result = apply_select_clause(result, select_fields)
    
    return result


def apply_where_clause(records, where_clause):
    """Apply WHERE filtering to records."""
    # Parse WHERE conditions (simple implementation supporting single conditions)
    # Format: field operator value
    match = re.match(r'(\w+)\s*([><=!]+)\s*(.+)', where_clause.strip())
    
    if not match:
        raise ValueError(f"Invalid WHERE clause format: {where_clause}")
    
    field, op, value = match.groups()
    value = value.strip()
    
    # Define operator mapping
    op_map = {
        '=': operator.eq,
        '==': operator.eq,
        '!=': operator.ne,
        '>': operator.gt,
        '>=': operator.ge,
        '<': operator.lt,
        '<=': operator.le
    }
    
    if op not in op_map:
        raise ValueError(f"Unsupported operator: {op}")
    
    op_func = op_map[op]
    
    # Try to convert value to appropriate type
    if value.startswith("'") and value.endswith("'"):
        # String value
        value = value[1:-1]
    else:
        # Try to convert to number
        try:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            # Keep as string if conversion fails
            pass
    
    # Filter records
    filtered_records = []
    for record in records:
        if field not in record:
            raise ValueError(f"Field '{field}' not found in records")
        
        record_value = record[field]
        try:
            if op_func(record_value, value):
                filtered_records.append(record)
        except TypeError:
            # Type mismatch in comparison
            raise ValueError(f"Cannot compare {type(record_value).__name__} with {type(value).__name__}")
    
    return filtered_records


def apply_order_by_clause(records, order_by_clause):
    """Apply ORDER BY sorting to records."""
    # Parse ORDER BY clause
    parts = order_by_clause.strip().split()
    field = parts[0]
    
    # Check for ASC/DESC
    descending = False
    if len(parts) > 1:
        if parts[1].upper() == 'DESC':
            descending = True
        elif parts[1].upper() != 'ASC':
            raise ValueError(f"Invalid ORDER BY direction: {parts[1]}")
    
    # Verify field exists
    if records and field not in records[0]:
        raise ValueError(f"Field '{field}' not found in records")
    
    # Sort records
    return sorted(records, key=lambda x: x.get(field), reverse=descending)


def apply_select_clause(records, select_fields):
    """Apply SELECT projection to records."""
    result = []
    
    for record in records:
        new_record = {}
        for field in select_fields:
            if field not in record:
                raise ValueError(f"Field '{field}' not found in records")
            new_record[field] = record[field]
        result.append(new_record)
    
    return result
