import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# === KROK 1: Ścieżka do Twojego pliku ===
# Zmień nazwę pliku na swoje zdjęcie (np. 'moje_oct.jpg')
IMAGE_PATH = './mk/MK_M05_20250108_144242_Anterior_Chamber_Radial_Full_Range_R_16mm_12288x6x3_scan1.png' 

try:
    img = mpimg.imread(IMAGE_PATH)
except FileNotFoundError:
    print(f"BŁĄD: Nie znaleziono pliku '{IMAGE_PATH}'. Wgraj zdjęcie do folderu ze skryptem.")
    # Tworzymy sztuczny obrazek testowy, żeby skrypt zadziałał demonstracyjnie
    img = np.zeros((800, 1200), dtype=np.uint8)
    # Rysujemy "oko" (uproszczone)
    cv2 = None
    try:
        import cv2
        img = np.zeros((800, 1200), dtype=np.uint8)
        cv2.ellipse(img, (600, 300), (500, 100), 0, 0, 180, 255, 5) # Rogówka
        cv2.line(img, (100, 300), (450, 450), 150, 15) # Tęczówka L
        cv2.line(img, (1100, 300), (750, 450), 150, 15) # Tęczówka P
        cv2.ellipse(img, (600, 550), (300, 50), 0, 180, 360, 100, 3) # Soczewka
    except ImportError:
        pass

# === KROK 2: Konfiguracja Etykiet (Tu wpiszesz swoje współrzędne) ===
# Format: "Nazwa": (X_celu, Y_celu, X_tekstu, Y_tekstu)
# X_celu, Y_celu -> Gdzie wskazuje grot strzałki
# X_tekstu, Y_tekstu -> Gdzie znajduje się napis
annotations = {
    "Iris":       (864, 1011,  870, 1200),
    "Cornea":       (3099, 78,  2900, 300),
    "Pupil":       (2020, 1008,  2000, 1200)
}

# === FUNKCJA POMOCNICZA: Znajdowanie współrzędnych ===
def on_click(event):
    if event.xdata is not None and event.ydata is not None:
        print(f"Kliknięto: X={int(event.xdata)}, Y={int(event.ydata)}")

# === RYSOWANIE FINALNE ===
def create_figure():
    # Ustawienia wykresu - wysoka jakość (DPI)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    
    # Wyświetlanie w skali szarości (medyczny standard OCT)
    if len(img.shape) == 3:
        ax.imshow(img)
    else:
        ax.imshow(img, cmap='gray')
    
    ax.axis('off') # Wyłączenie osi X/Y

    # Pętla rysująca strzałki
    for label, coords in annotations.items():
        x_target, y_target, x_text, y_text = coords
        
        # Stylizacja strzałki (Medyczny styl: biała/cyjan)
        ax.annotate(
            text=label,
            xy=(x_target, y_target),       # Grot strzałki
            xytext=(x_text, y_text),       # Pozycja tekstu
            color='white',                 # Kolor tekstu
            fontsize=12,                   # Rozmiar czcionki
            fontweight='bold',             # Pogrubienie
            arrowprops=dict(
                arrowstyle='->',           # Styl grota
                color='cyan',              # Kolor strzałki (dobry kontrast na OCT)
                lw=2,                      # Grubość linii
                connectionstyle="arc3,rad=0.2" # Lekkie zakrzywienie strzałki (ładniejsze)
            ),
            bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none", alpha=0.5) # Tło pod tekstem (dla czytelności)
        )

    plt.tight_layout()
    plt.title("Oznaczanie struktur na skanie OCT", color='white')
    
    # Podłączamy "klikacz" do znajdowania współrzędnych
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    
    print("--- INSTRUKCJA ---")
    print("1. Kliknij na obrazku, aby zobaczyć współrzędne X, Y w konsoli.")
    print("2. Przepisz te liczby do słownika 'annotations' w kodzie.")
    print("3. Zamknij okno, aby zakończyć.")
    plt.savefig('oct_labeled_final.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_figure()

