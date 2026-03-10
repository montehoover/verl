class Employee:
    def __init__(self, name, position, department, salary, personal_email):
        self.name = name
        self.position = position
        self.department = department
        self.salary = salary
        self.personal_email = personal_email

employees = [
    Employee("Alice Smith", "Software Engineer", "IT", 85000, "alice@personal.com"),
    Employee("Bob Johnson", "Project Manager", "IT", 95000, "bob@personal.com"),
    Employee("Carol Williams", "Data Analyst", "IT", 75000, "carol@personal.com")
]

def list_employees_by_department(department, format_template="{name} - {position}"):
    try:
        result = []
        found = False
        for employee in employees:
            if employee.department == department:
                formatted_string = format_template
                
                # Replace placeholders with actual values
                if "{name}" in formatted_string:
                    formatted_string = formatted_string.replace("{name}", employee.name)
                if "{position}" in formatted_string:
                    formatted_string = formatted_string.replace("{position}", employee.position)
                if "{department}" in formatted_string:
                    formatted_string = formatted_string.replace("{department}", employee.department)
                
                result.append(formatted_string)
                found = True
        
        if not found:
            return f"No employees found in department: {department}"
        
        return "\n".join(result)
            
    except Exception as e:
        raise ValueError(f"An error occurred while listing employees: {str(e)}")
