import os
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

class Contour:
    def __init__(self, slice_number, contour_number, points):
        self.slice_number = slice_number  # Номер среза
        self.contour_number = contour_number  # Номер контура
        self.points = points.astype(np.float64)  # Точки контура в формате double
        self.center, self.radius = self.calculate_average_center_and_radius(points)

    def calculate_average_center_and_radius(self, contour):
        if len(contour) == 0:
            return None, None

        points = np.squeeze(contour)  # Убираем лишние измерения

        if points.ndim == 1:
            points = np.expand_dims(points, axis=0)

        # Вычисляем средний центр
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        center = (center_x, center_y)

        # Вычисляем средний радиус
        radii = np.sqrt((points[:, 0] - center_x) ** 2 + (points[:, 1] - center_y) ** 2)
        average_radius = np.mean(radii)

        return center, average_radius

    def __repr__(self):
        return f"Contour(slice={self.slice_number}, contour={self.contour_number}, center={self.center}, radius={self.radius}, points={len(self.points)})"

def process_dicom_file(file_path, slice_number):
    """
    Обрабатывает DICOM-файл: извлекает контуры и создает объекты Contour.
    """
    dataset = pydicom.dcmread(file_path)
    pixel_array = dataset.pixel_array
    image = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX)
    image = np.uint8(image)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_objects = []
    for i, cnt in enumerate(contours):
        # Преобразуем точки контура в формат float64 (double)
        cnt_float64 = cnt.astype(np.float64)
        contour_obj = Contour(slice_number, i, cnt_float64)
        contour_objects.append(contour_obj)

    return image, contour_objects

def draw_contour(image, contour):
    """
    Рисует указанный контур на изображении.
    """
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Преобразуем точки контура обратно в int для отрисовки
    points_int = contour.points.astype(np.int32)
    cv2.drawContours(output_image, [points_int], -1, (0, 255, 0), 2)
    plt.imshow(output_image)
    plt.title(f"Contour {contour.contour_number} on Slice {contour.slice_number}")
    plt.axis('off')
    plt.show()

def find_closest_points(contour_points, reference_points, threshold=10):
    """
    Находит ближайшие точки в контуре к эталонным точкам.
    Если ошибка превышает порог, выводит точки контура.
    """
    closest_points = []
    for ref_point in reference_points:
        # Проверяем, что contour_points не пустой
        if len(contour_points) == 0:
            closest_points.append(ref_point)  # Если точек нет, возвращаем эталонную точку
            continue

        # Убедимся, что contour_points имеет правильную размерность (N x 2)
        if contour_points.ndim != 2 or contour_points.shape[1] != 2:
            print(f"Неправильная размерность точек контура: {contour_points.shape}")
            closest_points.append(ref_point)  # Возвращаем эталонную точку в случае ошибки
            continue

        # Вычисляем расстояния от эталонной точки до всех точек контура
        distances = np.linalg.norm(contour_points - ref_point, axis=1)
        # Находим индекс ближайшей точки
        closest_index = np.argmin(distances)
        closest_point = contour_points[closest_index]
        closest_points.append(closest_point)

    return closest_points

def plot_error_distances(all_contours, reference_contour_index, reference_slice_index, threshold=5):
    """
    Строит графики зависимости ошибки от номера файла для 0-й, 90-й, 180-й и последней точек.
    Если расстояние между эталонной точкой и точкой контура больше threshold, значение не сохраняется.
    """
    # Находим эталонный контур (80-й файл)
    reference_contour = None
    for contour in all_contours:
        if contour.slice_number == reference_slice_index and contour.contour_number == reference_contour_index:
            reference_contour = contour
            break

    if reference_contour is None:
        print("Эталонный контур не найден!")
        return

    # Выбираем 0-ю, 90-ю, 180-ю и последнюю точки эталонного контура
    num_points = len(reference_contour.points)
    reference_points = np.array([
        reference_contour.points[0],  # 0-я точка
        reference_contour.points[num_points // 4],  # 90-я точка
        reference_contour.points[num_points // 2],  # 180-я точка
        reference_contour.points[-1]  # Последняя точка
    ])

    # Убираем лишнее измерение с помощью np.squeeze
    reference_points = np.squeeze(reference_points)

    # Проверяем, что reference_points имеет правильную размерность (4 x 2)
    if reference_points.ndim != 2 or reference_points.shape[1] != 2:
        print(f"Неправильная размерность эталонных точек: {reference_points.shape}")
        return

    # Собираем расстояния для всех контуров
    distances_0 = []
    distances_90 = []
    distances_180 = []
    distances_last = []
    file_numbers = []

    # Создаем словарь для хранения контуров по номерам файлов
    contours_by_slice = {}
    for contour in all_contours:
        if contour.slice_number not in contours_by_slice:
            contours_by_slice[contour.slice_number] = []
        contours_by_slice[contour.slice_number].append(contour)

    # Обрабатываем каждый файл (срез)
    for slice_number in sorted(contours_by_slice.keys()):
        if slice_number == reference_slice_index:  # Исключаем эталонный файл
            continue

        # Берем первый контур в файле (или любой другой, если нужно)
        if len(contours_by_slice[slice_number]) == 0:
            print(f"В файле {slice_number} нет контуров!")
            continue

        contour = contours_by_slice[slice_number][0]  # Берем первый контур

        # Проверяем, что контур содержит точки
        if len(contour.points) == 0:
            print(f"Контур в файле {slice_number} не содержит точек!")
            continue

        # Убедимся, что точки контура имеют правильную размерность (N x 2)
        contour_points = np.squeeze(contour.points)  # Убираем лишнее измерение
        if contour_points.ndim != 2 or contour_points.shape[1] != 2:
            print(f"Неправильная размерность точек контура в файле {slice_number}: {contour_points.shape}")
            continue

        # Находим ближайшие точки в текущем контуре к эталонным точкам
        closest_points = find_closest_points(contour_points, reference_points, threshold)

        # Вычисляем расстояния до эталонных точек
        dist_0 = np.linalg.norm(closest_points[0] - reference_points[0])
        dist_90 = np.linalg.norm(closest_points[1] - reference_points[1])
        dist_180 = np.linalg.norm(closest_points[2] - reference_points[2])
        dist_last = np.linalg.norm(closest_points[3] - reference_points[3])

        # Фильтруем значения: сохраняем только те, где расстояние <= threshold
        if dist_0 <= threshold:
            distances_0.append(dist_0)
        if dist_90 <= threshold:
            distances_90.append(dist_90)
        if dist_180 <= threshold:
            distances_180.append(dist_180)
        if dist_last <= threshold:
            distances_last.append(dist_last)
        file_numbers.append(slice_number)

    # Строим графики
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(file_numbers[:len(distances_0)], distances_0, marker='o')
    plt.title("Ошибка для 0 градусов")
    plt.xlabel("Номер файла")
    plt.ylabel("Ошибка")

    plt.subplot(2, 2, 2)
    plt.plot(file_numbers[:len(distances_90)], distances_90, marker='o')
    plt.title("Ошибка для 90 градусов")
    plt.xlabel("Номер файла")
    plt.ylabel("Ошибка")

    plt.subplot(2, 2, 3)
    plt.plot(file_numbers[:len(distances_180)], distances_180, marker='o')
    plt.title("Ошибка для 180 градусов")
    plt.xlabel("Номер файла")
    plt.ylabel("Ошибка")

    plt.subplot(2, 2, 4)
    plt.plot(file_numbers[:len(distances_last)], distances_last, marker='o')
    plt.title("Ошибка для 270 градусов")
    plt.xlabel("Номер файла")
    plt.ylabel("Ошибка")

    plt.tight_layout()
    plt.show()

    # Сделать через угол

def print_first_point_of_contour(all_contours, contour_number):
    """
    Выводит первую точку заданного контура для каждого среза.

    :param all_contours: Список всех контуров (объекты Contour).
    :param contour_number: Номер контура, для которого нужно вывести первую точку.
    """
    # Создаем словарь для хранения контуров по номерам срезов
    contours_by_slice = {}
    for contour in all_contours:
        if contour.slice_number not in contours_by_slice:
            contours_by_slice[contour.slice_number] = []
        contours_by_slice[contour.slice_number].append(contour)

    # Проходим по каждому срезу
    for slice_number in sorted(contours_by_slice.keys()):
        # Ищем контур с заданным номером
        target_contour = None
        for contour in contours_by_slice[slice_number]:
            if contour.contour_number == contour_number:
                target_contour = contour
                break

        # Если контур найден, выводим первую точку
        if target_contour is not None:
            if len(target_contour.points) > 0:
                first_point = target_contour.points[0]  # Первая точка контура
                print(f"Срез {slice_number}: Первая точка контура {contour_number} = {first_point}")
            else:
                print(f"Срез {slice_number}: Контур {contour_number} не содержит точек.")
        else:
            print(f"Срез {slice_number}: Контур {contour_number} не найден.")

def main(dicom_directory):
    """
    Основная функция: обрабатывает все DICOM-файлы в директории.
    """
    all_contours = []
    for idx, filename in enumerate(os.listdir(dicom_directory)):
        if filename.endswith(".dcm"):
            file_path = os.path.join(dicom_directory, filename)
            image, contours = process_dicom_file(file_path, idx)
            all_contours.extend(contours)

    # Пример отображения первого контура
    if all_contours:
        print("Пример контура:")
        print(all_contours[1])  # Выводим информацию о первом контуре
        draw_contour(image, all_contours[1])  # Рисуем первый контур
        
        # Предположим, что all_contours уже заполнен данными
        print_first_point_of_contour(all_contours, contour_number=1)

        # Строим графики ошибок
        # Вызов функции с threshold=5
        plot_error_distances(all_contours, reference_contour_index=1, reference_slice_index=51, threshold=50)  # 80-й файл имеет индекс 52

if __name__ == "__main__":
    dicom_directory = "/home/beyong/Coding/coding/NIR/mri_clear"  # Укажите путь к вашим DICOM-файлам
    main(dicom_directory)