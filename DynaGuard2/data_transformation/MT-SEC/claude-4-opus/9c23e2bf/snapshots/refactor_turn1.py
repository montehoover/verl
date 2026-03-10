import re
import operator
from functools import partial

def run_custom_query(dataset, query):
    # Parse the query
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
    
    # Apply WHERE filtering
    result = dataset
    if where_conditions:
        ops = {
            '=': operator.eq,
            '!=': operator.ne,
            '<': operator.lt,
            '>': operator.gt,
            '<=': operator.le,
            '>=': operator.ge
        }
        
        filtered = []
        for record in result:
            include = True
            for field, op, value in where_conditions:
                if field not in record:
                    raise ValueError(f"Field '{field}' not found in dataset")
                if not ops[op](record[field], value):
                    include = False
                    break
            if include:
                filtered.append(record)
        result = filtered
    
    # Apply ORDER BY
    if order_field:
        if result and order_field not in result[0]:
            raise ValueError(f"Field '{order_field}' not found in dataset")
        result = sorted(result, key=lambda x: x.get(order_field), reverse=order_desc)
    
    # Apply SELECT projection
    if select_fields == ['*']:
        return result
    else:
        projected = []
        for record in result:
            new_record = {}
            for field in select_fields:
                if field not in record:
                    raise ValueError(f"Field '{field}' not found in dataset")
                new_record[field] = record[field]
            projected.append(new_record)
        return projected
