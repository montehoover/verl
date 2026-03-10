def generate_employee_summary(employee_id: str, template: str) -> str:
    try:
        # Access global variables
        global employees, performances
        
        if employee_id not in employees:
            raise ValueError("Employee not found")
        
        employee = employees[employee_id]
        performance = performances.get(employee_id)
        
        if not performance:
            raise ValueError("Performance data not found")
        
        # Check for sensitive information access attempts
        if 'feedback' in template or 'bonus' in template:
            raise ValueError("Access to sensitive information is not allowed")
        
        # Create a context dictionary for safe replacements
        context = {
            'employee.name': employee.name,
            'employee.position': employee.position,
            'employee.department': employee.department,
            'employee.emp_id': employee.emp_id,
            'performance.rating': str(performance.rating)
        }
        
        # Replace placeholders in the template
        result = template
        for placeholder, value in context.items():
            result = result.replace(f'{{{placeholder}}}', value)
        
        return result
        
    except Exception as e:
        raise ValueError(str(e))
