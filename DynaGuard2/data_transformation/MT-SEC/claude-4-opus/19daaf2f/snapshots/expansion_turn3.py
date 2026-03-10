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

def filter_data(records: List[Dict], condition: Callable[[Dict], bool]) -> List[Dict]:
    """
    Filter records based on a condition.
    
    Args:
        records: List of dictionaries representing records
        condition: A callable that takes a record and returns True if it meets the condition
        
    Returns:
        List of records that meet the condition
    """
    result = []
    
    for record in records:
        if condition(record):
            result.append(record)
    
    return result

def run_sql_query(dataset: List[Dict], sql_query: str) -> List[Dict]:
    """
    Process SQL-like commands on a dataset.
    
    Args:
        dataset: List of dictionaries representing the data
        sql_query: SQL-like query string with SELECT, WHERE, and ORDER BY support
        
    Returns:
        List of dictionaries containing query results
        
    Raises:
        ValueError: If query is malformed or execution fails
    """
    # Parse the SQL query
    sql_query = sql_query.strip()
    
    # Extract SELECT clause
    select_match = re.match(r'SELECT\s+(.*?)(?:\s+FROM|\s+WHERE|\s+ORDER\s+BY|$)', sql_query, re.IGNORECASE)
    if not select_match:
        raise ValueError("Malformed query: Missing SELECT clause")
    
    select_clause = select_match.group(1).strip()
    
    # Parse selected columns
    if select_clause == '*':
        selected_fields = None  # Select all fields
    else:
        selected_fields = [col.strip() for col in select_clause.split(',')]
    
    # Extract WHERE clause
    where_match = re.search(r'WHERE\s+(.*?)(?:\s+ORDER\s+BY|$)', sql_query, re.IGNORECASE)
    where_clause = where_match.group(1).strip() if where_match else None
    
    # Extract ORDER BY clause
    order_match = re.search(r'ORDER\s+BY\s+(.*?)(?:\s+ASC|\s+DESC|$)', sql_query, re.IGNORECASE)
    order_clause = order_match.group(1).strip() if order_match else None
    order_desc = bool(re.search(r'\s+DESC(?:\s|$)', sql_query, re.IGNORECASE))
    
    # Start with the full dataset
    result = dataset.copy()
    
    # Apply WHERE clause
    if where_clause:
        result = _apply_where_clause(result, where_clause)
    
    # Apply ORDER BY clause
    if order_clause:
        result = _apply_order_by(result, order_clause, order_desc)
    
    # Apply SELECT clause
    if selected_fields is not None:
        result = select_columns(result, selected_fields)
    
    return result

def _apply_where_clause(records: List[Dict], where_clause: str) -> List[Dict]:
    """Apply WHERE filtering to records."""
    # Parse comparison operators
    operators_map = {
        '>=': operator.ge,
        '<=': operator.le,
        '!=': operator.ne,
        '=': operator.eq,
        '>': operator.gt,
        '<': operator.lt,
    }
    
    # Find the operator and split the condition
    for op_str, op_func in operators_map.items():
        if op_str in where_clause:
            parts = where_clause.split(op_str, 1)
            if len(parts) == 2:
                field = parts[0].strip()
                value = parts[1].strip()
                
                # Remove quotes if present
                if value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                elif value.startswith('"') and value.endswith('"'):
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
                
                # Create condition function
                def condition(record, field=field, op_func=op_func, value=value):
                    if field not in record:
                        return False
                    record_value = record[field]
                    try:
                        # Try numeric comparison if possible
                        if isinstance(value, (int, float)) and not isinstance(record_value, str):
                            return op_func(float(record_value), value)
                        else:
                            return op_func(str(record_value), str(value))
                    except:
                        return False
                
                return filter_data(records, condition)
    
    raise ValueError(f"Malformed WHERE clause: {where_clause}")

def _apply_order_by(records: List[Dict], order_field: str, descending: bool = False) -> List[Dict]:
    """Apply ORDER BY sorting to records."""
    def get_sort_key(record):
        value = record.get(order_field, '')
        # Try to convert to number for proper sorting
        try:
            return float(value)
        except (ValueError, TypeError):
            return str(value)
    
    return sorted(records, key=get_sort_key, reverse=descending)
