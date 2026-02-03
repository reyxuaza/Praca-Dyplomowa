import cv2
import numpy as np
import sys 
import os 
import glob
import shutil

# =============================================================================
# 1. ALGORYTMY BIOMETRYCZNE
# =============================================================================

def log_gabor_filter(rows, cols, wavelength, orientation, sigma_on_f, dThetaOnSigma):
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    X = X - cols / 2
    Y = Y - rows / 2
    radius = np.sqrt(X**2 + Y**2)
    radius[int(rows/2), int(cols/2)] = 1.0 
    angle = np.arctan2(Y, X)
    log_gabor_radial = np.exp((-(np.log(radius * wavelength / cols))**2) / (2 * np.log(sigma_on_f)**2))
    log_gabor_radial[int(rows/2), int(cols/2)] = 0.0 
    theta = angle
    theta[theta < 0] = theta[theta < 0] + np.pi
    angle_diff = np.abs(theta - orientation)
    angle_diff = np.minimum(angle_diff, np.pi - angle_diff)
    log_gabor_angular = np.exp(-(angle_diff**2) / (2 * dThetaOnSigma**2))
    return (log_gabor_radial * log_gabor_angular).astype(np.float32)

def detect_specular_reflections(gray, pupil_center, pupil_radius):
    reflection_mask = np.zeros_like(gray, dtype=np.uint8)
    _, bright_thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    bright_thresh = cv2.dilate(bright_thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(bright_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            dist_to_pupil = np.sqrt((cx - pupil_center[0])**2 + (cy - pupil_center[1])**2)
            if area < 500 and dist_to_pupil < pupil_radius * 2:
                cv2.drawContours(reflection_mask, [contour], -1, 255, -1)
    return reflection_mask

def detect_eyelids(gray, pupil_center, pupil_radius, iris_radius):
    eyelid_mask = np.zeros_like(gray, dtype=np.uint8)
    roi_margin = int(iris_radius * 0.2) 
    y_min = max(0, pupil_center[1] - iris_radius - roi_margin)
    y_max = min(gray.shape[0], pupil_center[1] + iris_radius + roi_margin)
    x_min = max(0, pupil_center[0] - iris_radius - roi_margin)
    x_max = min(gray.shape[1], pupil_center[0] + iris_radius + roi_margin)
    roi = gray[y_min:y_max, x_min:x_max]
    edges = cv2.Canny(roi, 25, 80)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25, minLineLength=iris_radius//2, maxLineGap=15)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x1_full, y1_full = x1 + x_min, y1 + y_min
            x2_full, y2_full = x2 + x_min, y2 + y_min
            angle = abs(np.arctan2(y2_full - y1_full, x2_full - x1_full))
            if angle < np.pi/8 and y1_full < pupil_center[1] - pupil_radius * 0.2:
                y_line = int(max(y1_full, y2_full))
                eyelid_mask[0:y_line, x_min:x_max] = 255
            elif angle < np.pi/8 and y1_full > pupil_center[1] + pupil_radius * 0.2:
                y_line = int(min(y1_full, y2_full))
                eyelid_mask[y_line:, x_min:x_max] = 255
    return eyelid_mask

def detect_eyelashes(gray, pupil_center, iris_radius):
    kernel_size = 7
    mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
    mean_sq = cv2.blur((gray.astype(np.float32))**2, (kernel_size, kernel_size))
    variance = mean_sq - mean**2
    variance = np.sqrt(np.abs(variance))
    variance_norm = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, high_variance = cv2.threshold(variance_norm, 80, 255, cv2.THRESH_BINARY)
    _, dark_thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    eyelash_candidate = cv2.bitwise_and(high_variance, dark_thresh)
    kernel = np.ones((5,5), np.uint8)
    eyelash_candidate = cv2.morphologyEx(eyelash_candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
    pupil_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(pupil_mask, pupil_center, int(iris_radius * 0.5), 255, -1)
    return cv2.bitwise_and(eyelash_candidate, cv2.bitwise_not(pupil_mask))

# =============================================================================
# 2. DETEKCJA I NORMALIZACJA
# =============================================================================

def detect_pupil(gray):
    blurred = cv2.GaussianBlur(gray, (17, 17), 0)
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3), np.uint8)
    processed = cv2.dilate(cv2.erode(thresh, kernel, iterations=1), kernel, iterations=1)
    circles = cv2.HoughCircles(processed, cv2.HOUGH_GRADIENT, 1.2, 100, param1=100, param2=20, minRadius=10, maxRadius=80)
    if circles is not None:
        p = np.uint16(np.around(circles))[0, 0]
        return (p[0], p[1]), p[2]
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        ((x, y), r) = cv2.minEnclosingCircle(c)
        return (int(x), int(y)), int(r)
    return None, None

def detect_iris_non_concentric(gray, pupil_center, pupil_radius):
    max_r = min(gray.shape) // 2
    min_r = int(pupil_radius * 1.5)
    search_range = 10 
    step = 2
    best_score = -1
    best_circle = None 
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    radii_to_test = range(min_r, min(int(pupil_radius * 4), max_r), 2)
    for dx in range(-search_range, search_range + 1, step):
        for dy in range(-search_range, search_range + 1, step):
            cx = pupil_center[0] + dx
            cy = pupil_center[1] + dy
            if cx < 0 or cx >= gray.shape[1] or cy < 0 or cy >= gray.shape[0]: continue
            for r in radii_to_test:
                score = 0
                samples = 0
                for theta in np.linspace(0, 2*np.pi, 16, endpoint=False):
                    x_in = int(cx + (r - 2) * np.cos(theta))
                    y_in = int(cy + (r - 2) * np.sin(theta))
                    x_out = int(cx + (r + 2) * np.cos(theta))
                    y_out = int(cy + (r + 2) * np.sin(theta))
                    if 0 <= x_in < gray.shape[1] and 0 <= y_in < gray.shape[0] and \
                       0 <= x_out < gray.shape[1] and 0 <= y_out < gray.shape[0]:
                        score += (int(blurred[y_out, x_out]) - int(blurred[y_in, x_in]))
                        samples += 1
                if samples > 0 and (score/samples) > best_score:
                    best_score = score/samples
                    best_circle = (cx, cy, r)
    if best_circle: return (best_circle[0], best_circle[1]), best_circle[2]
    else: return pupil_center, int(pupil_radius * 2.5)

# =============================================================================
# 3. DYLACJA
# =============================================================================

def synthetic_dilation(image, dilation_factor=1.2, non_linearity=1.5):
    img_work = image.copy()
    if len(img_work.shape) == 3: gray = cv2.cvtColor(img_work, cv2.COLOR_BGR2GRAY)
    else: gray = img_work
    p_center, r_pupil = detect_pupil(gray)
    if p_center is None: return image
    i_center, r_iris = detect_iris_non_concentric(gray, p_center, r_pupil)
    cx, cy = p_center
    r_pupil_new = int(r_pupil * dilation_factor)
    if r_pupil_new >= r_iris: r_pupil_new = r_iris - 5
    rows, cols = gray.shape
    map_x, map_y = np.indices((rows, cols), dtype=np.float32)
    dx = map_y - cx
    dy = map_x - cy
    dist = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    r_src = dist.copy()
    mask_pupil = dist < r_pupil_new
    r_src[mask_pupil] = dist[mask_pupil] * (r_pupil / r_pupil_new)
    mask_iris = (dist >= r_pupil_new) & (dist < r_iris)
    if r_iris != r_pupil_new:
        u = (dist[mask_iris] - r_pupil_new) / (r_iris - r_pupil_new)
        v = np.power(u, non_linearity)
        r_src[mask_iris] = r_pupil + v * (r_iris - r_pupil)
    src_x = cx + r_src * np.cos(angle)
    src_y = cy + r_src * np.sin(angle)
    return cv2.remap(img_work, src_x, src_y, interpolation=cv2.INTER_LINEAR)

# =============================================================================
# 4. PRZETWARZANIE OBRAZU I HAMMING
# =============================================================================

def calculate_hamming_distance(code1, mask1, code2, mask2):
    if code1.shape != code2.shape: return 1.0
    c1 = (code1 > 128).astype(np.uint8)
    c2 = (code2 > 128).astype(np.uint8)
    m1 = (mask1 > 128).astype(np.uint8)
    m2 = (mask2 > 128).astype(np.uint8)
    max_shift = 24 
    best_hd = 1.0
    for shift in range(-max_shift, max_shift + 1):
        c2_shifted = np.roll(c2, shift, axis=1)
        m2_shifted = np.roll(m2, shift, axis=1)
        combined_mask = np.bitwise_and(m1, m2_shifted)
        total_bits = np.sum(combined_mask)
        if total_bits == 0: continue
        xor_result = np.bitwise_xor(c1, c2_shifted)
        mismatches = np.bitwise_and(xor_result, combined_mask)
        current_hd = np.sum(mismatches) / total_bits
        if current_hd < best_hd:
            best_hd = current_hd
    return best_hd

def process_iris_image(img, apply_dilation=False, dilation_factor=1.4, is_path=False, return_viz=False):
    """
    Dodano parametr return_viz. Jeśli True, zwraca (code, mask, visualization_image).
    """
    if is_path and isinstance(img, str):
        img_loaded = cv2.imread(img)
        if img_loaded is None: 
            return (None, None, None) if return_viz else (None, None)
    else:
        img_loaded = img
    
    if img_loaded is None: 
        return (None, None, None) if return_viz else (None, None)
    
    # Kopia do rysowania wizualizacji (kółek)
    viz_img = img_loaded.copy()

    if apply_dilation:
        img_loaded = synthetic_dilation(img_loaded, dilation_factor=dilation_factor)
        # Jeśli robimy dylację, wizualizacja też musi być na obrazie dentylowanym
        viz_img = img_loaded.copy() 

    gray = cv2.cvtColor(img_loaded, cv2.COLOR_BGR2GRAY)
    
    # 1. Detekcja
    pc, pr = detect_pupil(gray)
    if pc is None: 
        return (None, None, None) if return_viz else (None, None)
    
    ic, ir = detect_iris_non_concentric(gray, pc, pr)
    
    # Rysowanie na obrazie wizualizacji (Pupil=Czerwony, Iris=Zielony)
    cv2.circle(viz_img, pc, pr, (0, 0, 255), 2)
    cv2.circle(viz_img, ic, ir, (0, 255, 0), 2)
    cv2.circle(viz_img, pc, 2, (0, 0, 255), -1) # Środek źrenicy

    # 2. Maski
    ref_mask = detect_specular_reflections(gray, pc, pr)
    lid_mask = detect_eyelids(gray, pc, pr, ir)
    lash_mask = detect_eyelashes(gray, pc, ir)
    occ_mask = cv2.bitwise_or(ref_mask, cv2.bitwise_or(lid_mask, lash_mask))
    occ_mask = cv2.dilate(occ_mask, np.ones((7,7), np.uint8), iterations=1) 

    # 3. Normalizacja
    norm_w, norm_h = 512, 64
    map_x = np.zeros((norm_h, norm_w), dtype=np.float32)
    map_y = np.zeros((norm_h, norm_w), dtype=np.float32)
    diff_x, diff_y = pc[0] - ic[0], pc[1] - ic[1]
    
    for i in range(norm_w):
        theta = 2 * np.pi * (i / norm_w)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x_p = pc[0] + pr * cos_t
        y_p = pc[1] + pr * sin_t
        L_dot_D = diff_x * cos_t + diff_y * sin_t
        L_sq = diff_x**2 + diff_y**2
        delta = L_dot_D**2 - (L_sq - ir**2)
        if delta < 0: t_outer = ir 
        else: t_outer = -L_dot_D + np.sqrt(delta)
        for j in range(norm_h):
            r_norm = j / norm_h
            r_current = pr + r_norm * (t_outer - pr)
            map_x[j, i] = pc[0] + r_current * cos_t
            map_y[j, i] = pc[1] + r_current * sin_t

    iris_norm = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR)
    mask_norm = cv2.remap(occ_mask, map_x, map_y, cv2.INTER_NEAREST)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    iris_enhanced = clahe.apply(iris_norm)

    # 4. Kodowanie
    norm_float = iris_enhanced.astype(np.float32)
    rows, cols = norm_float.shape
    wavelengths = [12, 18, 24] 
    num_orientations = 4
    total_rows = rows * len(wavelengths)
    code = np.zeros((total_rows, cols * num_orientations * 2), dtype=np.uint8)
    code_mask = np.zeros_like(code)
    boundary_mask = np.zeros((rows, cols), dtype=np.uint8)
    boundary_mask[2:-2, :] = 255 
    
    row_offset = 0
    for wave in wavelengths:
        for k in range(num_orientations):
            orient = k * (np.pi / num_orientations)
            filt = log_gabor_filter(rows, cols, wave, orient, 0.55, 1.5)
            f_img = np.fft.fft2(norm_float)
            f_filt = f_img * np.fft.fftshift(filt)
            filtered = np.fft.ifft2(f_filt)
            real, imag = np.real(filtered), np.imag(filtered)
            mag = np.abs(filtered)
            threshold = np.percentile(mag, 20) * 1.5
            mask_fragile_real = (np.abs(real) < threshold)
            mask_fragile_imag = (np.abs(imag) < threshold)
            valid_mask = (mask_norm == 0).astype(np.uint8) * 255
            valid_mask = cv2.bitwise_and(valid_mask, boundary_mask)
            final_mask_real = valid_mask * (1 - mask_fragile_real.astype(np.uint8))
            final_mask_imag = valid_mask * (1 - mask_fragile_imag.astype(np.uint8))
            c_start = k * cols * 2
            r_start, r_end = row_offset, row_offset + rows
            code[r_start:r_end, c_start:c_start+cols] = (real > 0).astype(np.uint8) * 255
            code[r_start:r_end, c_start+cols:c_start+2*cols] = (imag > 0).astype(np.uint8) * 255
            code_mask[r_start:r_end, c_start:c_start+cols] = final_mask_real
            code_mask[r_start:r_end, c_start+cols:c_start+2*cols] = final_mask_imag
        row_offset += rows
    
    if return_viz:
        return code, code_mask, viz_img
    return code, code_mask

# =============================================================================
# 5. SKANOWANIE (3 TRYBY)
# =============================================================================

def process_dataset_pairs(root_dir):
    """Tryb 1: Przeszukuje strukturę folderów i zapisuje statystyki do pliku."""
    output_file = "najlepsze_pary_dylacja.txt"
    print(f"🔹 Rozpoczynam skanowanie struktury: {root_dir}")
    valid_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    with open(output_file, "w") as f_out:
        f_out.write("Folder_ID;ImageA;ImageB;HD_STD;HD_DILATED\n")
        
        for subject_id in range(1, 46):
            subject_str = str(subject_id)
            full_subject_path = os.path.join(root_dir, subject_str)
            if not os.path.isdir(full_subject_path): continue
            print(f"➡️ Analiza ID: {subject_id}...")

            for side in ['left', 'right']:
                side_path = os.path.join(full_subject_path, side)
                if not os.path.isdir(side_path): continue
                
                images_paths = []
                for ext in valid_exts:
                    images_paths.extend(glob.glob(os.path.join(side_path, ext)))
                if len(images_paths) < 2: continue

                # Cache
                cache = {}
                for img_p in images_paths:
                    name = os.path.basename(img_p)
                    try:
                        std_c, std_m = process_iris_image(img_p, apply_dilation=False, is_path=True)
                        dil_c, dil_m = process_iris_image(img_p, apply_dilation=True, dilation_factor=1.4, is_path=True)
                        if std_c is not None and dil_c is not None:
                            cache[name] = {'std': (std_c, std_m), 'dil': (dil_c, dil_m)}
                    except Exception: continue
                
                valid_names = list(cache.keys())
                if len(valid_names) < 2: continue

                # Znajdź najlepszą parę STD
                best_pair = (None, None)
                min_hd_std = 1.0
                for i in range(len(valid_names)):
                    for j in range(i + 1, len(valid_names)):
                        nA, nB = valid_names[i], valid_names[j]
                        cA, mA = cache[nA]['std']
                        cB, mB = cache[nB]['std']
                        hd = calculate_hamming_distance(cA, mA, cB, mB)
                        if hd < min_hd_std:
                            min_hd_std = hd
                            best_pair = (nA, nB)

                # Sprawdź dylację dla najlepszej pary
                if best_pair[0]:
                    wA, wB = best_pair
                    dA, _ = cache[wA]['dil']
                    dB, _ = cache[wB]['dil']
                    sA, _ = cache[wA]['std']
                    sB, _ = cache[wB]['std']
                    
                    # Cross-check
                    hd1 = calculate_hamming_distance(dA[0], dA[1], sB[0], sB[1]) # Dil A vs Std B
                    hd2 = calculate_hamming_distance(sA[0], sA[1], dB[0], dB[1]) # Std A vs Dil B
                    best_dil = min(hd1, hd2)
                    
                    line = f"{subject_id}_{side};{wA};{wB};{min_hd_std:.4f};{best_dil:.4f}\n"
                    f_out.write(line)
                    f_out.flush()
    print(f"\n Zakończono! Wyniki zapisano w: {output_file}")

def check_specific_folder(folder_path):
    """
    Tryb --check:
    1. Znajduje najlepszą parę.
    2. Generuje podgląd:
       [ SEGMENTACJA A ] [ SEGMENTACJA B ] [ SEGMENTACJA DILATED ]
       [ ORYGINAŁ A    ] [ ORYGINAŁ B    ] [ ORYGINAŁ DILATED    ]
    """
    output_dir = "wyniki_check"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print(f"🔹 SPRAWDZANIE SZCZEGÓŁOWE: {folder_path}")
    print(f"🔹 Folder wyjściowy: {output_dir}")
    
    valid_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    images_paths = []
    for ext in valid_exts:
        images_paths.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if len(images_paths) < 2:
        print("Za mało zdjęć w folderze (min. 2).")
        return

    # 1. Znajdź najlepszą parę w Standardzie
    print("Szukanie najlepszej pary w folderze...")
    best_hd = 1.0
    best_pair_files = (None, None)
    
    # Przechowujemy obrazy w pamięci
    loaded_imgs = {} 
    
    for i in range(len(images_paths)):
        pathA = images_paths[i]
        imgA = cv2.imread(pathA)
        if imgA is None: continue
        loaded_imgs[pathA] = imgA
        cA, mA = process_iris_image(imgA, apply_dilation=False, is_path=False)
        
        for j in range(i + 1, len(images_paths)):
            pathB = images_paths[j]
            imgB = cv2.imread(pathB)
            if imgB is None: continue
            
            cB, mB = process_iris_image(imgB, apply_dilation=False, is_path=False)
            
            if cA is not None and cB is not None:
                hd = calculate_hamming_distance(cA, mA, cB, mB)
                if hd < best_hd:
                    best_hd = hd
                    best_pair_files = (pathA, pathB)

    if best_pair_files[0] is None:
        print("❌ Nie udało się dopasować żadnej pary.")
        return

    pathA, pathB = best_pair_files
    nameA = os.path.basename(pathA)
    nameB = os.path.basename(pathB)
    print(f"Najlepsza para: {nameA} vs {nameB} (HD: {best_hd:.4f})")

    # 2. Sprawdź Dylację dla zwycięzców i pobierz WIZUALIZACJE (Viz)
    print("Generowanie wizualizacji segmentacji...")
    
    imgA = loaded_imgs[pathA]
    imgB = loaded_imgs[pathB]
    
    # Pobieramy: Kod, Maskę i Obrazek z kółkami (Viz)
    cA_std, mA_std, vizA = process_iris_image(imgA, apply_dilation=False, return_viz=True)
    cB_std, mB_std, vizB = process_iris_image(imgB, apply_dilation=False, return_viz=True)
    
    # Generujemy wersje dentylowane
    imgA_dil_raw = synthetic_dilation(imgA, dilation_factor=1.4)
    imgB_dil_raw = synthetic_dilation(imgB, dilation_factor=1.4)
    
    cA_dil, mA_dil, vizA_dil = process_iris_image(imgA_dil_raw, apply_dilation=False, return_viz=True)
    cB_dil, mB_dil, vizB_dil = process_iris_image(imgB_dil_raw, apply_dilation=False, return_viz=True)
    
    # Cross check HD
    hd_cross1 = calculate_hamming_distance(cA_dil, mA_dil, cB_std, mB_std) # Dil A vs Std B
    hd_cross2 = calculate_hamming_distance(cA_std, mA_std, cB_dil, mB_dil) # Std A vs Dil B
    
    # Wybieramy zwycięzcę dylacji do wyświetlenia
    img_viz_Dil_Row1 = None # Segmentacja
    img_viz_Dil_Row2 = None # Surowy/Dentylowany
    caption_dil = ""
    
    if hd_cross1 < hd_cross2:
        print(f"   Lepszy wariant: Dilated A vs Std B -> HD: {hd_cross1:.4f}")
        img_viz_Dil_Row1 = vizA_dil     # Segmentacja dentylowana A
        img_viz_Dil_Row2 = imgA_dil_raw # Surowa dentylowana A
        caption_dil = f"Dilated A vs Std B: {hd_cross1:.4f}"
    else:
        print(f"   Lepszy wariant: Std A vs Dilated B -> HD: {hd_cross2:.4f}")
        img_viz_Dil_Row1 = vizB_dil     # Segmentacja dentylowana B
        img_viz_Dil_Row2 = imgB_dil_raw # Surowa dentylowana B
        caption_dil = f"Std A vs Dilated B: {hd_cross2:.4f}"

    # 3. Budowanie Dużego Obrazu (Kolażu)
    
    # Dodaj napisy na Row 1 (Segmentacja)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thick = 2
    
    # Upewnij się, że mamy obrazy
    if vizA is None or vizB is None or img_viz_Dil_Row1 is None:
        print("Błąd generowania wizualizacji.")
        return

    # Kopiujemy, żeby nie pisać po oryginałach w pamięci
    r1_c1 = vizA.copy()
    r1_c2 = vizB.copy()
    r1_c3 = img_viz_Dil_Row1.copy()
    
    r2_c1 = imgA.copy()
    r2_c2 = imgB.copy()
    r2_c3 = img_viz_Dil_Row2.copy()

    # Napisy nagłówkowe na obrazkach
    cv2.putText(r1_c1, "Segmentation A", (10, 30), font, scale, (0, 255, 255), thick)
    cv2.putText(r1_c2, "Segmentation B", (10, 30), font, scale, (0, 255, 255), thick)
    cv2.putText(r1_c3, "Segmentation DIL", (10, 30), font, scale, (0, 0, 255), thick)
    
    cv2.putText(r2_c1, "Original A", (10, 30), font, scale, (0, 255, 255), thick)
    cv2.putText(r2_c2, "Original B", (10, 30), font, scale, (0, 255, 255), thick)
    cv2.putText(r2_c3, "Simulated DIL", (10, 30), font, scale, (0, 0, 255), thick)

    # Sklejanie wierszy (HSTACK)
    row1 = np.hstack((r1_c1, r1_c2, r1_c3))
    row2 = np.hstack((r2_c1, r2_c2, r2_c3))
    
    # Sklejanie całości (VSTACK)
    # Stopka tekstowa
    footer_height = 80
    footer = np.zeros((footer_height, row1.shape[1], 3), dtype=np.uint8)
    
    text_std = f"Best Standard Pair: {nameA} vs {nameB} = {best_hd:.4f}"
    text_dil = f"Dilated Test: {caption_dil}"
    
    cv2.putText(footer, text_std, (20, 35), font, 0.7, (255, 255, 255), 1)
    cv2.putText(footer, text_dil, (20, 70), font, 0.7, (100, 255, 100), 2)
    
    # Jeśli obrazy są grayscale, konwersja do BGR dla vstack
    if len(row1.shape) == 2: row1 = cv2.cvtColor(row1, cv2.COLOR_GRAY2BGR)
    if len(row2.shape) == 2: row2 = cv2.cvtColor(row2, cv2.COLOR_GRAY2BGR)
    
    final_collage = np.vstack((row1, row2, footer))
    
    save_path = os.path.join(output_dir, "podsumowanie_wizualne.jpg")
    cv2.imwrite(save_path, final_collage)
    print(f"Zapisano kolaż: {save_path}")
# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie:")
        print("  1. Skanowanie folderów: python3 allcheck.py <folder_data>")
        print("  2. Podgląd folderu:     python3 allcheck.py --check <sciezka_do_folderu>")
        sys.exit(1)

    arg1 = sys.argv[1]

    if arg1 == "--check":
        if len(sys.argv) < 3:
            print("Podaj ścieżkę po fladze --check")
        else:
            check_specific_folder(sys.argv[2])
    elif os.path.isdir(arg1):
        process_dataset_pairs(arg1)
    else:
        print("Błąd: Nieprawidłowa ścieżka.")
