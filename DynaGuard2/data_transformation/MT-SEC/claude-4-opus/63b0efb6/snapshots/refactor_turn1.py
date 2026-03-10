import re
import operator
from functools import partial

def run_sql_query(records, command):
    # Parse the SQL-like command
    command = command.strip()
    
    # Extract SELECT clause
    select_match = re.match(r'SELECT\s+(.+?)(?:\s+FROM|\s+WHERE|\s+ORDER\s+BY|$)', command, re.IGNORECASE)
    if not select_match:
        raise ValueError("Invalid query format: missing SELECT clause")
    
    select_clause = select_match.group(1).strip()
    
    # Extract WHERE clause if present
    where_match = re.search(r'WHERE\s+(.+?)(?:\s+ORDER\s+BY|$)', command, re.IGNORECASE)
    where_clause = where_match.group(1).strip() if where_match else None
    
    # Extract ORDER BY clause if present
    order_match = re.search(r'ORDER\s+BY\s+(.+?)(?:\s+ASC|\s+DESC|$)', command, re.IGNORECASE)
    order_clause = order_match.group(1).strip() if order_match else None
    
    # Check if ASC or DESC is specified
    desc_match = re.search(r'ORDER\s+BY\s+.+?\s+DESC', command, re.IGNORECASE)
    ascending = not bool(desc_match)
    
    # Process SELECT clause
    if select_clause == '*':
        selected_fields = None  # Select all fields
    else:
        selected_fields = [field.strip() for field in select_clause.split(',')]
    
    # Apply WHERE clause
    filtered_records = records
    if where_clause:
        filtered_records = []
        for record in records:
            if evaluate_where_clause(record, where_clause):
                filtered_records.append(record)
    
    # Apply ORDER BY clause
    if order_clause:
        try:
            filtered_records = sorted(filtered_records, 
                                    key=lambda x: x.get(order_clause, ''), 
                                    reverse=not ascending)
        except Exception:
            raise ValueError(f"Cannot order by field: {order_clause}")
    
    # Apply SELECT clause
    result = []
    for record in filtered_records:
        if selected_fields is None:
            result.append(record.copy())
        else:
            selected_record = {}
            for field in selected_fields:
                if field in record:
                    selected_record[field] = record[field]
                else:
                    raise ValueError(f"Field '{field}' not found in records")
            result.append(selected_record)
    
    return result

def evaluate_where_clause(record, where_clause):
    # Handle AND conditions
    and_conditions = re.split(r'\s+AND\s+', where_clause, flags=re.IGNORECASE)
    
    for condition in and_conditions:
        # Handle OR conditions within each AND group
        or_conditions = re.split(r'\s+OR\s+', condition, flags=re.IGNORECASE)
        or_result = False
        
        for or_condition in or_conditions:
            # Parse individual condition
            match = re.match(r'(\w+)\s*(=|!=|<>|<=|>=|<|>)\s*(.+)', or_condition.strip())
            if not match:
                raise ValueError(f"Invalid WHERE condition: {or_condition}")
            
            field, op, value = match.groups()
            field = field.strip()
            value = value.strip()
            
            # Remove quotes if present
            if (value.startswith("'") and value.endswith("'")) or \
               (value.startswith('"') and value.endswith('"')):
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
            
            # Get record value
            if field not in record:
                raise ValueError(f"Field '{field}' not found in record")
            
            record_value = record[field]
            
            # Evaluate condition
            if op == '=':
                result = record_value == value
            elif op == '!=' or op == '<>':
                result = record_value != value
            elif op == '<':
                result = record_value < value
            elif op == '>':
                result = record_value > value
            elif op == '<=':
                result = record_value <= value
            elif op == '>=':
                result = record_value >= value
            else:
                raise ValueError(f"Unknown operator: {op}")
            
            if result:
                or_result = True
                break
        
        if not or_result:
            return False
    
    return True
