import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import svgwrite  # Для сохранения в SVG


def calculate_average_center_and_radius(contour):
    if len(contour) == 0:
        return None, None

    points = np.squeeze(contour)

    if points.ndim == 1:
        points = np.expand_dims(points, axis=0)

    center_x = np.mean(points[:, 0])
    center_y = np.mean(points[:, 1])
    center = (center_x, center_y)

    radii = np.sqrt((points[:, 0] - center_x) ** 2 + (points[:, 1] - center_y) ** 2)

    average_radius = np.mean(radii)

    return center, average_radius

def translate(x, y, center):
    return x - center[0], y - center[1]

def rotate(x, y, angle):
    matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    product = np.dot(matrix, np.array([[x], [y]]))
    return product.ravel()[0], product.ravel()[1]

def sort_by_distance_with_radius(points, reference_point):
    """
    Сортирует массив точек с радиусами в порядке увеличения расстояния от заданной точки.
    
    :param points: Список элементов [(x, y), radius], где (x, y) — координаты точки, radius — радиус.
    :param reference_point: Кортеж (x, y), заданная точка для расчета расстояния.
    :return: Отсортированный список элементов [(x, y), radius].
    """
    def distance(point1, point2):
        # Вычисление Евклидова расстояния
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    # Сортировка по расстоянию до reference_point
    sorted_points = sorted(points, key=lambda item: distance(item[0], reference_point))
    return sorted_points


def sort_by_min_distance_and_radius(array1, array2):
    """
    Сортирует второй массив по минимальному расстоянию между координатами и радиусом,
    чтобы соответствовать элементам первого массива.

    :param array1: Список элементов [(x, y), radius] — первый массив.
    :param array2: Список элементов [(x, y), radius] — второй массив.
    :return: Отсортированный второй массив.
    """
    def distance(point1, point2):
        # Вычисление Евклидова расстояния между двумя точками
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def match_score(item1, item2):
        # Вычисление "оценки соответствия" по расстоянию и разнице радиусов
        coord_distance = distance(item1[0], item2[0])
        radius_difference = abs(item1[1] - item2[1])
        return coord_distance + radius_difference  # Можно добавить веса, если расстояние или радиус важнее
    
    # Создаём копию второго массива, чтобы исключать элементы при сопоставлении
    remaining_array2 = array2[:]
    sorted_array2 = []

    for item1 in array1:
        # Ищем элемент из второго массива с минимальной "оценкой соответствия"
        best_match = min(remaining_array2, key=lambda item2: match_score(item1, item2))
        sorted_array2.append(best_match)
        remaining_array2.remove(best_match)  # Удаляем элемент, чтобы он не повторялся

    return sorted_array2

def calculate_errors(array1, array2):
    """
    Вычисляет ошибки для координат и радиусов между двумя массивами.
    
    :param array1: Список элементов [(x, y), radius] — первый массив (эталонный).
    :param array2: Список элементов [(x, y), radius] — второй массив (проверяемый).
    :return: Два списка: ошибки для координат и ошибки для радиусов.
    """
    coordinate_errors = []
    radius_errors = []

    def distance(point1, point2):
        # Вычисление Евклидова расстояния между двумя точками
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    # Проверяем, что массивы одинаковой длины
    if len(array1) != len(array2):
        raise ValueError("Массивы должны быть одинаковой длины")

    for item1, item2 in zip(array1, array2):
        coord1, radius1 = item1
        coord2, radius2 = item2

        # Вычисление ошибки координат (расстояние между точками)
        coord_error = distance(coord1, coord2)
        coordinate_errors.append(coord_error)


        # Вычисление ошибки радиуса (модуль разности радиусов)
        radius_error = abs(radius1 - radius2)
        radius_errors.append(radius_error)

    return coordinate_errors, radius_errors

# Путь к DICOM-файлу
dicom_file_path = '/home/beyong/Coding/coding/NIR/mri_clear/1.2.840.113619.2.312.4120.8418826.11464.1715661998.653.dcm'

# Чтение DICOM-файла
dataset = pydicom.dcmread(dicom_file_path)

# Извлечение пиксельных данных
pixel_array = dataset.pixel_array

image = 255 * (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array))
image = image.astype(np.uint8)

# Нормализация пиксельных значений к диапазону 0-255
image = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX)
image = np.uint8(image)

# Опционально: Применение сглаживания для снижения шума
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Применение порогового преобразования (Оцу)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Поиск контуров
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Создание копии изображения для рисования контуров
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Рисование контуров
# cv2.drawContours(output_image, contours, -1, (0, 255, 0), 1)

centers_and_radii = []
radiuses = []
centers = []

i = 0 # Счетчик контуров

for cnt in contours:
    center, avg_radius = calculate_average_center_and_radius(cnt)

    if center is None or avg_radius is None:
        continue

    buf = int(avg_radius)
    radiuses.append(avg_radius)
    centers_and_radii.append((center, avg_radius))
    avg_radius = int(avg_radius)
    centers.append(center)
    i += 1

    buf = (int(center[0]), int(center[1]))
    cv2.circle(output_image, buf, avg_radius, (0, 0, 255), 1)


# Создание SVG изображения
width, height = image.shape[1], image.shape[0]
dwg = svgwrite.Drawing('contours_with_circles.svg', size=(width, height))

# Добавляем контуры как многоугольники в SVG
for cnt in contours:
    points = cnt[:, 0, :].tolist()
    dwg.add(dwg.polygon(points, fill='none', stroke='black'))

# Добавляем окружности по центрам и радиусам
for center, radius in centers_and_radii:
        dwg.add(dwg.rect(insert=center, size=(1, 1), fill="red"))

for center, radius in centers_and_radii:
    dwg.add(dwg.circle(center, r=radius, stroke='red', fill='none'))

def find_max_radius(centers_and_radii):
    max_radius = 0
    centers = [0, 0]
    for center, radius in centers_and_radii:
        if radius > max_radius:
            max_radius = radius
            max_center = center
    return max_radius, max_center

max_radius, max_center = find_max_radius(centers_and_radii)
print(f"Максимальный радиус: {max_radius}", f"Центр: {max_center}")
x, y = max_center
small_step = 21.2
medium_step1, medium_step2 = 26.5, 10.6
large_step = 53
# Отсюда все точки
dwg.add(dwg.circle(center=(x, y), r=91, stroke='green', fill='none'))
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

for (x, y), radius in small_circles:
    dwg.add(dwg.circle(center=(x, y), r=1.59, stroke='green', fill='none'))
    dwg.add(dwg.circle(center=(x, y), r=0.1, stroke='green', fill='none'))

for (x, y), radius in medium_circles:
    dwg.add(dwg.circle(center=(x, y), r=3.127, stroke='green', fill='none'))
    dwg.add(dwg.circle(center=(x, y), r=0.1, stroke='green', fill='none'))

for (x, y), radius in large_circles:
    dwg.add(dwg.circle(center=(x, y), r=5.3, stroke='green', fill='none'))
    dwg.add(dwg.circle(center=(x, y), r=0.1, stroke='green', fill='none'))
# Сохранение SVG файла
dwg.save()

print("SVG файл сохранен с контурами и окружностями.")
print(len(radiuses))
x, y = max_center
acc_sorted_points = sort_by_distance_with_radius(all_circles, (x, y))

nonacc_sorted_points = sort_by_min_distance_and_radius(acc_sorted_points, centers_and_radii)

for point in acc_sorted_points:
        print(f"Accurate: {point}")

for point in nonacc_sorted_points:
        print(f"Nonaccurate: {point}")

xy_errors, radius_errors = [], []
xy_errors, radius_errors = calculate_errors(acc_sorted_points, nonacc_sorted_points)


y_axis = []
for i in range (0, len(xy_errors)):
    y_axis.append(i)

plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.plot(y_axis, radius_errors, label='Radius error', marker="o")
# plt.title('Ошибка для радиуса')
#
# plt.subplot(1, 2, 2)
# plt.plot(y_axis, xy_errors, label='X or Y error', marker="o")
# plt.title('Ошибка для координат')

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Исходное изображение')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output_image)
plt.title('Изображение с контурами и окружностями')
plt.axis('off')
plt.savefig('output.png')

plt.show()
