import re
import operator
from functools import partial

def handle_sql_query(records, sql_command):
    """
    Process a custom SQL-like query on data represented as a list of dictionaries.
    
    Args:
        records: A list of dictionaries where each dictionary represents a record in the dataset
        sql_command: A string containing the SQL-like query to be executed
        
    Returns:
        A list containing dictionaries that represent the results of the executed query
        
    Raises:
        ValueError: Raised if the query is not correctly formed or execution fails
    """
    if not sql_command or not isinstance(sql_command, str):
        raise ValueError("Invalid SQL command")
    
    # Normalize the SQL command
    sql_command = ' '.join(sql_command.split())
    
    # Parse SELECT clause
    select_match = re.search(r'SELECT\s+(.*?)(?:\s+FROM|\s+WHERE|\s+ORDER\s+BY|$)', sql_command, re.IGNORECASE)
    if not select_match:
        raise ValueError("Invalid query: SELECT clause not found")
    
    select_fields = [field.strip() for field in select_match.group(1).split(',')]
    
    # Parse WHERE clause
    where_match = re.search(r'WHERE\s+(.*?)(?:\s+ORDER\s+BY|$)', sql_command, re.IGNORECASE)
    where_conditions = where_match.group(1) if where_match else None
    
    # Parse ORDER BY clause
    order_match = re.search(r'ORDER\s+BY\s+(.*?)(?:\s+ASC|\s+DESC|$)', sql_command, re.IGNORECASE)
    order_field = order_match.group(1).strip() if order_match else None
    order_desc = bool(re.search(r'\s+DESC', sql_command, re.IGNORECASE)) if order_match else False
    
    # Apply WHERE filtering
    filtered_records = records
    if where_conditions:
        filtered_records = apply_where_clause(records, where_conditions)
    
    # Apply SELECT projection
    if select_fields == ['*']:
        result_records = [dict(record) for record in filtered_records]
    else:
        result_records = []
        for record in filtered_records:
            selected_record = {}
            for field in select_fields:
                if field not in record:
                    raise ValueError(f"Field '{field}' not found in records")
                selected_record[field] = record[field]
            result_records.append(selected_record)
    
    # Apply ORDER BY sorting
    if order_field:
        if order_field not in (result_records[0] if result_records else {}):
            if result_records:
                raise ValueError(f"Field '{order_field}' not found in records")
        result_records.sort(key=lambda x: x.get(order_field), reverse=order_desc)
    
    return result_records

def apply_where_clause(records, where_conditions):
    """Apply WHERE clause filtering to records."""
    # Parse conditions - supports simple comparisons and AND/OR
    conditions = parse_where_conditions(where_conditions)
    
    filtered = []
    for record in records:
        if evaluate_conditions(record, conditions):
            filtered.append(record)
    
    return filtered

def parse_where_conditions(where_str):
    """Parse WHERE clause conditions into a structured format."""
    # Handle AND/OR operators
    if ' AND ' in where_str.upper():
        parts = re.split(r'\s+AND\s+', where_str, flags=re.IGNORECASE)
        return ('AND', [parse_single_condition(part) for part in parts])
    elif ' OR ' in where_str.upper():
        parts = re.split(r'\s+OR\s+', where_str, flags=re.IGNORECASE)
        return ('OR', [parse_single_condition(part) for part in parts])
    else:
        return parse_single_condition(where_str)

def parse_single_condition(condition_str):
    """Parse a single condition like 'field = value'."""
    # Match various operators
    match = re.match(r'(\w+)\s*(=|!=|<>|<=|>=|<|>)\s*(.+)', condition_str.strip())
    if not match:
        raise ValueError(f"Invalid WHERE condition: {condition_str}")
    
    field, op, value = match.groups()
    
    # Clean up value - remove quotes if present
    value = value.strip()
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        value = value[1:-1]
    else:
        # Try to convert to number
        try:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass
    
    # Map operators
    op_map = {
        '=': operator.eq,
        '!=': operator.ne,
        '<>': operator.ne,
        '<': operator.lt,
        '<=': operator.le,
        '>': operator.gt,
        '>=': operator.ge
    }
    
    return (field, op_map[op], value)

def evaluate_conditions(record, conditions):
    """Evaluate conditions against a record."""
    if isinstance(conditions, tuple) and len(conditions) == 3:
        # Single condition
        field, op_func, value = conditions
        if field not in record:
            raise ValueError(f"Field '{field}' not found in records")
        return op_func(record[field], value)
    
    # Compound condition
    logic_op, sub_conditions = conditions
    if logic_op == 'AND':
        return all(evaluate_conditions(record, cond) for cond in sub_conditions)
    elif logic_op == 'OR':
        return any(evaluate_conditions(record, cond) for cond in sub_conditions)
    
    raise ValueError("Invalid condition structure")
