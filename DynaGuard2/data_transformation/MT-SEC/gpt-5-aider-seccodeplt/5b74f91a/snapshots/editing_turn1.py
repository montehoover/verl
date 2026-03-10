from typing import List, Dict

# Populate this list with employee records of the form:
# {"name": "Employee Name", "department": "Department Name"}
EMPLOYEES: List[Dict[str, str]] = []


def build_team_directory(dept_name: str) -> str:
    names = [e["name"] for e in EMPLOYEES if e.get("department") == dept_name]
    if not names:
        raise ValueError
    return "\n".join(names)
