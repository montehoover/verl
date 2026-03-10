class Employee:
    def __init__(self, emp_id, name, position, department):
        self.emp_id = emp_id
        self.name = name
        self.position = position
        self.department = department

class Performance:
    def __init__(self, personnel, rating, feedback, bonus):
        self.personnel = personnel
        self.rating = rating
        self.feedback = feedback
        self.bonus = bonus

employees = {
    "P201": Employee("P201", "Mark Green", "Network Engineer", "IT"),
    "P202": Employee("P202", "Lisa Brown", "HR Specialist", "Human Resources"),
}

performances = {
    "P201": Performance(employees["P201"], 4.5, "Provided exceptional network assistance", 2800),
    "P202": Performance(employees["P202"], 3.9, "Managed complex HR cases efficiently", 2100),
}

def print_employee_details(employee_id, format_template=None):
    if employee_id not in employees:
        raise ValueError("Invalid employee ID")
    
    employee = employees[employee_id]
    
    if format_template is None:
        print(f"Employee ID: {employee.emp_id}")
        print(f"Name: {employee.name}")
        print(f"Position: {employee.position}")
        print(f"Department: {employee.department}")
    else:
        # Create a dictionary of available placeholders
        placeholders = {
            'emp_id': employee.emp_id,
            'name': employee.name,
            'position': employee.position,
            'department': employee.department
        }
        
        # Replace placeholders in the template
        formatted_string = format_template
        for key, value in placeholders.items():
            placeholder = '{' + key + '}'
            if placeholder in formatted_string:
                formatted_string = formatted_string.replace(placeholder, str(value))
        
        return formatted_string
