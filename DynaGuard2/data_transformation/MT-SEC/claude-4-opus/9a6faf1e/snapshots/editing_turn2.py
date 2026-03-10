import requests

def read_text_file(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except:
        return ""
