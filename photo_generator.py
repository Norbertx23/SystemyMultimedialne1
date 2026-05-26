import requests

width = 512
height = 512
save_path = "."

for i in range(3):
    url = f"https://picsum.photos/{width}/{height}"
    response = requests.get(url)

    if response.status_code == 200:
        with open(f"{save_path}/image{i}.jpg", "wb") as file:
            file.write(response.content)
        print(f"Zdjęcie {width}x{height} zostało pomyślnie pobrane!")
    else:
        print("Błąd podczas pobierania zdjęcia.")