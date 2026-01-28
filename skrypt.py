import os
from PIL import Image

def convert_bmp_to_png(source_folder, delete_original=False):
    """
    Konwertuje wszystkie pliki BMP w folderze na PNG.
    
    Args:
        source_folder (str): Ścieżka do folderu z obrazami.
        delete_original (bool): Czy usuwać pliki BMP po konwersji (True/False).
    """
    
    # Sprawdzenie czy folder istnieje
    if not os.path.exists(source_folder):
        print(f"Błąd: Folder '{source_folder}' nie istnieje.")
        return

    converted_count = 0

    # Przejście przez pliki w folderze
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(".bmp"):
            bmp_path = os.path.join(source_folder, filename)
            
            # Tworzenie nowej nazwy z rozszerzeniem .png
            name_without_ext = os.path.splitext(filename)[0]
            png_path = os.path.join(source_folder, name_without_ext + ".png")

            try:
                # Otwarcie i zapis jako PNG
                with Image.open(bmp_path) as img:
                    img.save(png_path, "PNG")
                
                print(f"Przekonwertowano: {filename} -> {name_without_ext}.png")
                converted_count += 1

                # Opcjonalne usuwanie oryginału
                if delete_original:
                    os.remove(bmp_path)
                    print(f"Usunięto oryginał: {filename}")

            except Exception as e:
                print(f"Błąd przy pliku {filename}: {e}")

    print(f"\nZakończono. Przekonwertowano plików: {converted_count}")

# --- KONFIGURACJA ---
folder_do_zdjec = "./kk"  # "." oznacza folder, w którym jest ten skrypt
usun_bmp = False       # Zmień na True, jeśli chcesz od razu kasować pliki BMP

convert_bmp_to_png(folder_do_zdjec, delete_original=usun_bmp)
