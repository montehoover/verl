from collections import defaultdict
from typing import Callable, List, Dict
import re
import operator
from functools import partial

def select_fields(records, fields):
    """
    Extract specific fields from a list of dictionaries.
    
    Args:
        records: List of dictionaries representing data records
        fields: List of field names (strings) to extract
        
    Returns:
        List of dictionaries containing only the specified fields
    """
    result = []
    
    for record in records:
        selected_record = {}
        for field in fields:
            if field in record:
                selected_record[field] = record[field]
        result.append(selected_record)
    
    return result


def filter_data(records: List[Dict], condition: Callable[[Dict], bool]) -> List[Dict]:
    """
    Filter records based on a given condition.
    
    Args:
        records: List of dictionaries representing data records
        condition: A callable that takes a record and returns True if it should be included
        
    Returns:
        List of dictionaries that satisfy the condition
    """
    return [record for record in records if condition(record)]


def execute_query_cmd(dataset_list: List[Dict], sql_query: str) -> List[Dict]:
    """
    Execute SQL-like queries on the dataset.
    
    Args:
        dataset_list: List of dictionaries representing the dataset
        sql_query: SQL-like query string (supports SELECT, WHERE, ORDER BY)
        
    Returns:
        List of dictionaries with query results
        
    Raises:
        ValueError: If the query is malformed or fails
    """
    try:
        # Parse the SQL-like query
        query_upper = sql_query.upper()
        
        # Extract SELECT clause
        select_match = re.search(r'SELECT\s+(.+?)(?:\s+FROM|\s+WHERE|\s+ORDER\s+BY|$)', sql_query, re.IGNORECASE)
        if not select_match:
            raise ValueError("Malformed query: SELECT clause not found")
        
        select_clause = select_match.group(1).strip()
        
        # Parse fields to select
        if select_clause == '*':
            fields_to_select = None
        else:
            fields_to_select = [field.strip() for field in select_clause.split(',')]
        
        # Start with all records
        result = dataset_list.copy()
        
        # Extract and apply WHERE clause
        where_match = re.search(r'WHERE\s+(.+?)(?:\s+ORDER\s+BY|$)', sql_query, re.IGNORECASE)
        if where_match:
            where_clause = where_match.group(1).strip()
            
            # Parse WHERE conditions (simplified - supports single conditions with =, >, <, >=, <=, !=)
            condition_pattern = r'(\w+)\s*(=|!=|>=|<=|>|<)\s*(.+?)(?:\s+AND|\s+OR|$)'
            conditions = re.findall(condition_pattern, where_clause, re.IGNORECASE)
            
            if not conditions:
                raise ValueError("Malformed WHERE clause")
            
            # Apply conditions (simplified - only supports single condition or AND logic)
            for field, op, value in conditions:
                # Clean up value
                value = value.strip().strip("'\"")
                
                # Get the operator function
                op_map = {
                    '=': operator.eq,
                    '!=': operator.ne,
                    '>': operator.gt,
                    '<': operator.lt,
                    '>=': operator.ge,
                    '<=': operator.le
                }
                
                op_func = op_map.get(op)
                if not op_func:
                    raise ValueError(f"Unsupported operator: {op}")
                
                # Try to convert value to appropriate type
                try:
                    # Try numeric conversion first
                    if '.' in value:
                        typed_value = float(value)
                    else:
                        typed_value = int(value)
                except ValueError:
                    # Keep as string
                    typed_value = value
                
                # Filter records
                def condition_func(record, field=field, op_func=op_func, typed_value=typed_value):
                    if field not in record:
                        return False
                    try:
                        return op_func(record[field], typed_value)
                    except:
                        # Type mismatch - try string comparison
                        return op_func(str(record[field]), str(typed_value))
                
                result = filter_data(result, condition_func)
        
        # Extract and apply ORDER BY clause
        order_match = re.search(r'ORDER\s+BY\s+(\w+)(?:\s+(ASC|DESC))?', sql_query, re.IGNORECASE)
        if order_match:
            order_field = order_match.group(1)
            order_direction = order_match.group(2) if order_match.group(2) else 'ASC'
            
            reverse = order_direction.upper() == 'DESC'
            
            # Sort the results
            def get_sort_key(record):
                return record.get(order_field, '')
            
            result = sorted(result, key=get_sort_key, reverse=reverse)
        
        # Apply field selection
        if fields_to_select:
            result = select_fields(result, fields_to_select)
        
        return result
        
    except Exception as e:
        raise ValueError(f"Query execution failed: {str(e)}")
