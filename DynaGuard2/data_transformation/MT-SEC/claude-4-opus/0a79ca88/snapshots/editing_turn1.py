import requests

TRUSTED_USER_PROFILES = {}

def get_user_details(u_id):
    # Fetch user profile from remote service
    response = requests.get(f"https://api.userservice.com/profiles/{u_id}")
    
    if response.status_code == 200:
        profile_data = response.json()
        return profile_data
    else:
        return None
