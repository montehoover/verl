recent_titles = []

def create_announcement_html(title, content, notes):
    global recent_titles
    recent_titles.append(title)
    
    html = f"<h1>{title}</h1>\n"
    html += f"<p>{content}</p>\n"
    html += f"<p>{notes}</p>\n"
    html += "<h2>Recent Titles:</h2>\n<ul>\n"
    for recent_title in recent_titles:
        html += f"  <li><b>{recent_title}</b></li>\n"
    html += "</ul>"
    
    return html
