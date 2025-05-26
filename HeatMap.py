import cv2
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import os
from glob import glob
from scipy.optimize import linear_sum_assignment

def calculate_average_center_and_radius(contour):
    contour = np.squeeze(contour)
    if contour.ndim != 2:
        return None, None

    x = contour[:, 0]
    y = contour[:, 1]
    x_m = np.mean(x)
    y_m = np.mean(y)
    r = np.sqrt((x - x_m) ** 2 + (y - y_m) ** 2)
    r_m = np.mean(r)
    return (int(x_m), int(y_m)), r_m

def find_max_radius(centers_and_radii):
    if not centers_and_radii:
        return 0, (0, 0)
    max_center, max_radius = max(centers_and_radii, key=lambda x: x[1])
    return max_radius, max_center

def sort_by_distance_with_radius(points, center):
    cx, cy = center
    return sorted(points, key=lambda p: np.hypot(p[0][0] - cx, p[0][1] - cy) + p[1])


def sort_by_min_distance_and_radius(reference_points, measured_points):
    # Строим матрицу стоимости для сопоставления
    cost_matrix = []
    for ref in reference_points:
        (ref_x, ref_y), ref_r = ref
        row = []
        for meas in measured_points:
            (meas_x, meas_y), meas_r = meas
            dist = np.hypot(ref_x - meas_x, ref_y - meas_y) + abs(ref_r - meas_r)
            row.append(dist)
        cost_matrix.append(row)

    # Применяем метод Венгера для поиска минимальной стоимости
    cost_matrix = np.array(cost_matrix)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Создаем итоговый список сопоставленных точек
    result = [(measured_points[i][0][0], measured_points[i][0][1], measured_points[i][1]) for i in col_ind]
    return result


def calculate_errors(reference_points, measured_points):
    xy_errors = []
    radius_errors = []
    for ref, meas in zip(reference_points, measured_points):
        (ref_x, ref_y), ref_r = ref
        meas_x, meas_y, meas_r = meas
        xy_error = np.hypot(ref_x - meas_x, ref_y - meas_y)
        radius_error = abs(ref_r - meas_r)
        xy_errors.append(xy_error)
        radius_errors.append(radius_error)
    return xy_errors, radius_errors


# Окружности по шаблону
x, y = 127.5, 127.5
small_step = 21.2
medium_step1, medium_step2 = 26.5, 10.6
large_step = 53

the_one = [(x , y), 91]

small_circles = [
    [(x + small_step, y + small_step), 1.59],
    [(x + small_step, y), 1.59],
    [(x, y + small_step), 1.59],
    [(x - small_step, y - small_step), 1.59],
    [(x, y - small_step), 1.59],
    [(x - small_step, y), 1.59],
    [(x + small_step, y - small_step), 1.59],
    [(x - small_step, y + small_step), 1.59],
    [(x + 2*small_step, y + 2*small_step), 1.59],
    [(x + 2*small_step, y), 1.59],
    [(x, y + 2*small_step), 1.59],
    [(x - 2*small_step, y - 2*small_step), 1.59],
    [(x, y - 2*small_step), 1.59],
    [(x - 2*small_step, y), 1.59],
    [(x + 2*small_step, y - 2*small_step), 1.59],
    [(x - 2*small_step, y + 2*small_step), 1.59],
    [(x + small_step, y + 2*small_step), 1.59],
    [(x + 2*small_step, y + small_step), 1.59],
    [(x - small_step, y - 2*small_step), 1.59],
    [(x - 2*small_step, y - small_step), 1.59],
    [(x + small_step, y - 2*small_step), 1.59],
    [(x + 2*small_step, y - small_step), 1.59],
    [(x - small_step, y + 2*small_step), 1.59],
    [(x - 2*small_step, y + small_step), 1.59],
    [(x - 2*small_step, y - 3*small_step), 1.59],
    [(x - small_step, y - 3*small_step), 1.59],
    [(x, y - 3*small_step), 1.59],
    [(x + small_step, y - 3*small_step), 1.59],
    [(x + 2*small_step, y - 3*small_step), 1.59],
    [(x - 2*small_step, y + 3*small_step), 1.59],
    [(x - small_step, y + 3*small_step), 1.59],
    [(x, y + 3*small_step), 1.59],
    [(x + small_step, y + 3*small_step), 1.59],
    [(x + 2*small_step, y + 3*small_step), 1.59],
    [(x + 3*small_step, y - 2*small_step), 1.59],
    [(x + 3*small_step, y - small_step), 1.59],
    [(x + 3*small_step, y), 1.59],
    [(x + 3*small_step, y + small_step), 1.59],
    [(x + 3*small_step, y + 2*small_step), 1.59],
    [(x - 3*small_step, y - 2*small_step), 1.59],
    [(x - 3*small_step, y - small_step), 1.59],
    [(x - 3*small_step, y), 1.59],
    [(x - 3*small_step, y + small_step), 1.59],
    [(x - 3*small_step, y + 2*small_step), 1.59]
]

medium_circles = [
    [(x + large_step, y + large_step), 3.127],
    [(x - large_step, y + large_step), 3.127],
    [(x + large_step, y - large_step), 3.127],
    [(x - large_step, y - large_step), 3.127]
]

large_circles = [
    [(x + medium_step1, y - medium_step2), 5.3],
    [(x + medium_step1 + 15.9, y - medium_step2), 5.3],
    [(x + medium_step1 + 47.7, y - medium_step2), 5.3],
    [(x + medium_step2, y + medium_step1), 5.3],
    [(x + medium_step2, y + medium_step1 + 15.9), 5.3],
    [(x + medium_step2, y + medium_step1 + 47.7), 5.3],
    [(x - medium_step1, y + medium_step2), 5.3],
    [(x - medium_step1 - 15.9, y + medium_step2), 5.3],
    [(x - medium_step1 - 47.7, y + medium_step2), 5.3],
    [(x - medium_step2, y - medium_step1), 5.3],
    [(x - medium_step2, y - medium_step1 - 15.9), 5.3],
    [(x - medium_step2, y - medium_step1 - 47.7), 5.3]
]


       
all_circles = [the_one] + small_circles + medium_circles + large_circles

# Папка с DICOM-файлами
dicom_folder = '/home/beyong/Coding/coding/NIR/mri_clear'
dicom_files = sorted(glob(os.path.join(dicom_folder, '*.dcm')))

# Массивы для накопления ошибок
all_xy_errors = []
all_radius_errors = []

for slice_idx, dicom_file_path in enumerate(dicom_files):
    print(f"Обработка файла {slice_idx + 1}/{len(dicom_files)}: {dicom_file_path}")
    dataset = pydicom.dcmread(dicom_file_path)
    pixel_array = dataset.pixel_array

    # Нормализация изображения
    image = 255 * (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array))
    image = image.astype(np.uint8)

    image = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX)
    image = np.uint8(image)

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    centers_and_radii = []
    for cnt in contours:
        center, avg_radius = calculate_average_center_and_radius(cnt)
        if center is None or avg_radius is None:
            continue
        centers_and_radii.append((center, avg_radius))

    # Вывод количества окружностей
    print(f"Количество окружностей в срезе {slice_idx + 1}: {len(centers_and_radii)}")

    if not centers_and_radii:
        continue

    max_radius, max_center = find_max_radius(centers_and_radii)
    center = (127.5, 127.5)

    # Сопоставление и ошибки
    acc_sorted_points = sort_by_distance_with_radius(all_circles, center)
    nonacc_sorted_points = sort_by_min_distance_and_radius(acc_sorted_points, centers_and_radii)
    xy_errors, radius_errors = calculate_errors(acc_sorted_points, nonacc_sorted_points)

    all_xy_errors.append(xy_errors)
    all_radius_errors.append(radius_errors)
    
    # --- Визуализация ---
    debug_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Нарисовать найденные окружности (зелёные)
    for (cx, cy), r in centers_and_radii:
        cv2.circle(debug_image, (int(cx), int(cy)), int(r), (0, 255, 0), 1)
        cv2.circle(debug_image, (int(cx), int(cy)), 1, (0, 255, 0), -1)

    # Нарисовать эталонные окружности (синие)
    for (rx, ry), rr in acc_sorted_points:
        cv2.circle(debug_image, (int(rx), int(ry)), int(rr), (255, 0, 0), 1)
        cv2.circle(debug_image, (int(rx), int(ry)), 1, (255, 0, 0), -1)

    # Нарисовать линии между эталонной и найденной окружностью (жёлтые)
    for ref, meas in zip(acc_sorted_points, nonacc_sorted_points):
        (rx, ry), _ = ref
        mx, my, _ = meas
        cv2.line(debug_image, (int(rx), int(ry)), (int(mx), int(my)), (0, 255, 255), 1)

    # Сохранить отладочное изображение
    debug_path = f"debug_output/slice_{slice_idx:03d}_circles.png"
    os.makedirs("debug_output", exist_ok=True)
    cv2.imwrite(debug_path, debug_image)
    print(f"Сохранено изображение среза {slice_idx + 1} с окружностями: {debug_path}")




# Преобразование в numpy для построения карты ошибок
all_xy_errors = np.array(all_xy_errors)

all_radius_errors = np.array(all_radius_errors)

def smooth_errors(errors):
    smoothed = []
    for i in range(len(errors) - 2):  # Останавливаемся за 2 элемента до конца
        window = errors[i:i+3]        # Берём 3 последовательных значения
        avg = sum(window) / 3        # Вычисляем среднее
        smoothed.append(avg)         # Добавляем в результат
    return smoothed

all_xy_errors = smooth_errors(all_xy_errors)
all_radius_errors = smooth_errors(all_radius_errors)

plt.figure(figsize=(18, 8))
plt.subplot(1, 2, 1)
img = plt.imshow(all_xy_errors, cmap='viridis', aspect='auto', vmin=0, vmax=np.percentile(all_xy_errors, 95))
plt.colorbar(img, label="Координатная ошибка (пиксели)")
plt.xlabel("Номер окружности")
plt.ylabel("Номер среза")
plt.title("Тепловая карта ошибок координат")

# Аналогично для второй карты
plt.subplot(1, 2, 2)
img2 = plt.imshow(all_radius_errors, cmap='viridis', aspect='auto')
plt.colorbar(img2, label="Ошибка радиуса (пиксели)")
plt.xlabel("Номер окружности")
plt.title("Тепловая карта ошибок радиуса")

plt.tight_layout()
plt.show()
