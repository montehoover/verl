from collections import defaultdict
from typing import Callable, List, Dict
import re
import operator
from functools import partial

def select_columns(records, field_names):
    """
    Extract specific fields from a list of records.
    
    Args:
        records: List of dictionaries representing records
        field_names: List of field names (strings) to extract
        
    Returns:
        List of dictionaries containing only the specified fields
    """
    result = []
    
    for record in records:
        selected_record = {}
        for field in field_names:
            if field in record:
                selected_record[field] = record[field]
        result.append(selected_record)
    
    return result

def apply_filter(records: List[Dict], condition: Callable[[Dict], bool]) -> List[Dict]:
    """
    Filter records based on a condition function.
    
    Args:
        records: List of dictionaries representing records
        condition: A callable that takes a record and returns True if it should be included
        
    Returns:
        List of dictionaries that meet the condition
    """
    return [record for record in records if condition(record)]

def process_sql_request(dataset_records: List[Dict], sql_statement: str) -> List[Dict]:
    """
    Process SQL-like commands including SELECT, WHERE, and ORDER BY clauses.
    
    Args:
        dataset_records: List of dictionaries representing the dataset
        sql_statement: SQL-like query string
        
    Returns:
        List of dictionaries containing query results
        
    Raises:
        ValueError: If the query is malformed or fails
    """
    # Parse the SQL statement
    sql_statement = sql_statement.strip()
    
    # Extract SELECT clause
    select_match = re.match(r'^SELECT\s+(.*?)\s+FROM', sql_statement, re.IGNORECASE)
    if not select_match:
        raise ValueError("Malformed query: missing SELECT or FROM clause")
    
    select_fields = select_match.group(1).strip()
    
    # Handle SELECT *
    if select_fields == '*':
        selected_fields = None
    else:
        selected_fields = [field.strip() for field in select_fields.split(',')]
    
    # Extract WHERE clause if present
    where_match = re.search(r'WHERE\s+(.*?)(?:\s+ORDER\s+BY|$)', sql_statement, re.IGNORECASE)
    where_condition = None
    if where_match:
        where_clause = where_match.group(1).strip()
        where_condition = parse_where_clause(where_clause)
    
    # Extract ORDER BY clause if present
    order_match = re.search(r'ORDER\s+BY\s+(.*?)(?:\s+ASC|\s+DESC|$)', sql_statement, re.IGNORECASE)
    order_field = None
    order_desc = False
    if order_match:
        order_field = order_match.group(1).strip()
        if re.search(r'\s+DESC', sql_statement, re.IGNORECASE):
            order_desc = True
    
    # Apply operations
    result = dataset_records
    
    # Apply WHERE filter
    if where_condition:
        result = apply_filter(result, where_condition)
    
    # Apply ORDER BY
    if order_field:
        try:
            result = sorted(result, key=lambda x: x.get(order_field, ''), reverse=order_desc)
        except Exception:
            raise ValueError(f"Failed to order by field: {order_field}")
    
    # Apply SELECT
    if selected_fields is not None:
        result = select_columns(result, selected_fields)
    
    return result

def parse_where_clause(where_clause: str) -> Callable[[Dict], bool]:
    """
    Parse WHERE clause and return a condition function.
    
    Args:
        where_clause: WHERE clause string
        
    Returns:
        A callable that evaluates the condition for a record
        
    Raises:
        ValueError: If the WHERE clause is malformed
    """
    # Simple parser for conditions like "field = value" or "field > value"
    # Support for =, !=, <, >, <=, >=
    
    # Try to match comparison operators
    match = re.match(r'^(\w+)\s*(=|!=|<|>|<=|>=)\s*(.+)$', where_clause.strip())
    if not match:
        raise ValueError(f"Malformed WHERE clause: {where_clause}")
    
    field, op, value = match.groups()
    value = value.strip()
    
    # Remove quotes if present
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
    
    # Map operators to functions
    op_map = {
        '=': operator.eq,
        '!=': operator.ne,
        '<': operator.lt,
        '>': operator.gt,
        '<=': operator.le,
        '>=': operator.ge
    }
    
    op_func = op_map.get(op)
    if not op_func:
        raise ValueError(f"Unsupported operator: {op}")
    
    def condition(record: Dict) -> bool:
        if field not in record:
            return False
        record_value = record[field]
        try:
            # Try to convert record value to same type as comparison value
            if isinstance(value, (int, float)):
                if isinstance(record_value, str):
                    if '.' in record_value:
                        record_value = float(record_value)
                    else:
                        record_value = int(record_value)
            return op_func(record_value, value)
        except:
            # If conversion fails, compare as strings
            return op_func(str(record_value), str(value))
    
    return condition
