import requests
from PIL import Image
import io

width = 512
height = 512
save_path = "/Users/norbertswistak/SystemyMultimedialne1/lab11/"
amount = 1

save_as_binary = True

for i in range(amount):
    url = f"https://picsum.photos/{width}/{height}"
    response = requests.get(url)

    if response.status_code == 200:
        if save_as_binary:
            image_data = io.BytesIO(response.content)
            img = Image.open(image_data)
            binary_img = img.convert('L').point(lambda p: 255 if p > 128 else 0).convert('1')

            file_name = f"{save_path}/image_binary_{i}.png"
            binary_img.save(file_name)
            print(f"Binarny obraz {width}x{height} zapisany jako {file_name}!")

        else:
            file_name = f"{save_path}/image_normal_{i}.jpg"
            with open(file_name, "wb") as file:
                file.write(response.content)
            print(f"Normalne zdjęcie {width}x{height} zapisane jako {file_name}!")

    else:
        print(f"Błąd podczas pobierania zdjęcia {i}. Kod błędu: {response.status_code}")