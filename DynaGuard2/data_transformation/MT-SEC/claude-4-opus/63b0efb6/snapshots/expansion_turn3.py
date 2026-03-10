from collections import defaultdict
from typing import Callable, List, Dict
import re
import operator
from functools import partial

def select_fields(records, field_names):
    """
    Extract specific fields from a list of records.
    
    Args:
        records: List of dictionaries representing records
        field_names: List of field names to extract
        
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
    Filter records based on a given condition.
    
    Args:
        records: List of dictionaries representing records
        condition: A callable that takes a record and returns True if it should be included
        
    Returns:
        List of dictionaries that satisfy the condition
    """
    return [record for record in records if condition(record)]


def run_sql_query(records: List[Dict], command: str) -> List[Dict]:
    """
    Execute SQL-like queries on records.
    
    Args:
        records: List of dictionaries representing records
        command: SQL-like query string supporting SELECT, WHERE, and ORDER BY
        
    Returns:
        List of dictionaries with query results
        
    Raises:
        ValueError: If the query is malformed or fails
    """
    try:
        # Parse SELECT clause
        select_match = re.search(r'SELECT\s+(.*?)(?:\s+WHERE|\s+ORDER\s+BY|$)', command, re.IGNORECASE)
        if not select_match:
            raise ValueError("Query must start with SELECT")
        
        select_clause = select_match.group(1).strip()
        
        # Handle SELECT *
        if select_clause == '*':
            selected_fields = None
        else:
            # Parse comma-separated fields
            selected_fields = [field.strip() for field in select_clause.split(',')]
        
        # Start with all records
        result = records.copy()
        
        # Parse WHERE clause
        where_match = re.search(r'WHERE\s+(.*?)(?:\s+ORDER\s+BY|$)', command, re.IGNORECASE)
        if where_match:
            where_clause = where_match.group(1).strip()
            
            # Parse simple conditions (field operator value)
            condition_match = re.match(r'(\w+)\s*(=|!=|<|>|<=|>=)\s*(.+)', where_clause)
            if not condition_match:
                raise ValueError(f"Invalid WHERE clause: {where_clause}")
            
            field, op, value = condition_match.groups()
            field = field.strip()
            value = value.strip()
            
            # Try to parse value as number
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                # Remove quotes if present
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
            
            # Map operator strings to operator functions
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
            
            # Filter records
            filtered = []
            for record in result:
                if field in record:
                    try:
                        if op_func(record[field], value):
                            filtered.append(record)
                    except:
                        # Skip records where comparison fails
                        pass
            result = filtered
        
        # Parse ORDER BY clause
        order_match = re.search(r'ORDER\s+BY\s+(\w+)(?:\s+(ASC|DESC))?', command, re.IGNORECASE)
        if order_match:
            order_field = order_match.group(1)
            order_dir = order_match.group(2) or 'ASC'
            
            # Sort records
            reverse = order_dir.upper() == 'DESC'
            result = sorted(result, key=lambda x: x.get(order_field, ''), reverse=reverse)
        
        # Apply field selection if not SELECT *
        if selected_fields is not None:
            result = select_fields(result, selected_fields)
        
        return result
        
    except Exception as e:
        raise ValueError(f"Query failed: {str(e)}")
