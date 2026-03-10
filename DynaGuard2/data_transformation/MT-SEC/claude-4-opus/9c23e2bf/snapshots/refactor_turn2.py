import re
import operator
from functools import partial

def parse_query(query):
    """Parse SQL-like query into structured components."""
    query = query.strip()
    
    # Extract SELECT clause
    select_match = re.match(r'SELECT\s+(.+?)(?:\s+FROM|\s+WHERE|\s+ORDER\s+BY|$)', query, re.IGNORECASE)
    if not select_match:
        raise ValueError("Invalid query format: SELECT clause not found")
    
    select_fields = [field.strip() for field in select_match.group(1).split(',')]
    
    # Extract WHERE clause
    where_match = re.search(r'WHERE\s+(.+?)(?:\s+ORDER\s+BY|$)', query, re.IGNORECASE)
    where_conditions = []
    if where_match:
        where_clause = where_match.group(1)
        # Parse WHERE conditions (supports =, !=, <, >, <=, >=)
        condition_pattern = r'(\w+)\s*(=|!=|<|>|<=|>=)\s*([\'"]?)(.+?)\3'
        conditions = re.findall(condition_pattern, where_clause)
        for field, op, quote, value in conditions:
            # Convert value to appropriate type
            if not quote:  # No quotes, try to convert to number
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            where_conditions.append((field, op, value))
    
    # Extract ORDER BY clause
    order_match = re.search(r'ORDER\s+BY\s+(\w+)(?:\s+(ASC|DESC))?', query, re.IGNORECASE)
    order_field = None
    order_desc = False
    if order_match:
        order_field = order_match.group(1)
        order_desc = order_match.group(2) and order_match.group(2).upper() == 'DESC'
    
    return {
        'select': select_fields,
        'where': where_conditions,
        'order_by': order_field,
        'order_desc': order_desc
    }

def apply_where_filter(dataset, conditions):
    """Apply WHERE conditions to filter dataset."""
    if not conditions:
        return dataset
    
    ops = {
        '=': operator.eq,
        '!=': operator.ne,
        '<': operator.lt,
        '>': operator.gt,
        '<=': operator.le,
        '>=': operator.ge
    }
    
    filtered = []
    for record in dataset:
        include = True
        for field, op, value in conditions:
            if field not in record:
                raise ValueError(f"Field '{field}' not found in dataset")
            if not ops[op](record[field], value):
                include = False
                break
        if include:
            filtered.append(record)
    return filtered

def apply_order_by(dataset, order_field, order_desc):
    """Apply ORDER BY to sort dataset."""
    if not order_field:
        return dataset
    
    if dataset and order_field not in dataset[0]:
        raise ValueError(f"Field '{order_field}' not found in dataset")
    
    return sorted(dataset, key=lambda x: x.get(order_field), reverse=order_desc)

def apply_select_projection(dataset, select_fields):
    """Apply SELECT projection to limit fields in result."""
    if select_fields == ['*']:
        return dataset
    
    projected = []
    for record in dataset:
        new_record = {}
        for field in select_fields:
            if field not in record:
                raise ValueError(f"Field '{field}' not found in dataset")
            new_record[field] = record[field]
        projected.append(new_record)
    return projected

def execute_query_pipeline(dataset, parsed_query):
    """Execute parsed query components in pipeline fashion."""
    result = dataset
    result = apply_where_filter(result, parsed_query['where'])
    result = apply_order_by(result, parsed_query['order_by'], parsed_query['order_desc'])
    result = apply_select_projection(result, parsed_query['select'])
    return result

def run_custom_query(dataset, query):
    """Execute a SQL-like query on a dataset."""
    parsed_query = parse_query(query)
    return execute_query_pipeline(dataset, parsed_query)
