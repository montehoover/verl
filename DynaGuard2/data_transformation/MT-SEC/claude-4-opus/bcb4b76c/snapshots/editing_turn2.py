def forum_post_with_points(title, content, points):
    print(f"<h2>{title}</h2>")
    print(f"<p>{content}</p>")
    print("Discussion points:")
    for point in points:
        print(f"- {point}")
