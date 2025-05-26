import os
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

class Contour:
    def __init__(self, slice_number, contour_number, points):
        self.slice_number = slice_number
        self.contour_number = contour_number
        self.points = points.astype(np.float32)
        self.center, self.radius = self.calculate_average_center_and_radius(self.points)
        self.area = cv2.contourArea(self.points.reshape(-1, 1, 2).astype(np.float32))

    def calculate_average_center_and_radius(self, contour):
        points = contour.reshape(-1, 2)
        center_x, center_y = np.mean(points[:, 0]), np.mean(points[:, 1])
        radii = np.sqrt((points[:, 0] - center_x)**2 + (points[:, 1] - center_y)**2)
        return (center_x, center_y), np.mean(radii)

    def find_intersection_points(self, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        center_x, center_y = self.center
        
        # Направляющий вектор линии
        dir_x = np.cos(angle_rad)
        dir_y = np.sin(angle_rad)
        
        # Создаем длинную линию для пересечения с контуром
        line_length = max(self.radius * 3, 100)  # Достаточно длинная линия
        line_p1 = (center_x - line_length * dir_x, center_y - line_length * dir_y)
        line_p2 = (center_x + line_length * dir_x, center_y + line_length * dir_y)
        
        # Преобразуем контур в формат для пересечения
        contour_points = self.points.reshape(-1, 2).astype(np.int32)
        
        # Создаем изображение для поиска пересечений
        img_size = int(max(contour_points.max(axis=0)) * 1.2)
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.drawContours(mask, [contour_points], -1, 255, 1)
        
        # Рисуем линию
        line_mask = np.zeros_like(mask)
        cv2.line(line_mask, 
                (int(line_p1[0]), int(line_p1[1])),
                (int(line_p2[0]), int(line_p2[1])),
                255, 1)
        
        # Находим пересечения
        intersections = cv2.bitwise_and(mask, line_mask)
        y, x = np.where(intersections > 0)
        
        # Если точек пересечения мало, пробуем увеличить толщину линии
        if len(x) < 2:
            line_mask = np.zeros_like(mask)
            cv2.line(line_mask, 
                    (int(line_p1[0]), int(line_p1[1])),
                    (int(line_p2[0]), int(line_p2[1])),
                    255, 2)
            intersections = cv2.bitwise_and(mask, line_mask)
            y, x = np.where(intersections > 0)
        
        if len(x) >= 2:
            # Берем первую и последнюю точки пересечения
            points = list(zip(x, y))
            # Сортируем по расстоянию от центра вдоль линии
            points.sort(key=lambda p: (p[0]-center_x)*dir_x + (p[1]-center_y)*dir_y)
            return [points[0], points[-1]]
        
        return None

def process_dicom_file(file_path, slice_number):
    """Обрабатывает DICOM-файл и извлекает контуры"""
    try:
        dataset = pydicom.dcmread(file_path)
        pixel_array = dataset.pixel_array
        
        # Нормализация изображения
        image = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX)
        image = np.uint8(image)
        
        # Улучшение изображения перед поиском контуров
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Поиск контуров
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_objects = []
        for i, cnt in enumerate(contours):
            if len(cnt) < 5:  # Пропускаем слишком маленькие контуры
                continue
                
            cnt_processed = cnt.astype(np.float32)
            contour_obj = Contour(slice_number, i, cnt_processed)
            contour_objects.append(contour_obj)
            
        return image, contour_objects
        
    except Exception as e:
        print(f"Ошибка при обработке файла {file_path}: {str(e)}")
        return None, []

def draw_contour_with_fixed_angle_line(image, contour, angle_deg):
    """Улучшенная визуализация с диагностикой"""
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    points_int = contour.points.astype(np.int32)
    
    # Рисуем контур
    cv2.drawContours(output_image, [points_int], -1, (0, 255, 0), 2)
    
    # Рисуем центр
    center_int = (int(contour.center[0]), int(contour.center[1]))
    cv2.circle(output_image, center_int, 3, (255, 0, 0), -1)
    
    # Находим точки пересечения
    intersections = contour.find_intersection_points(angle_deg)
    
    if intersections and len(intersections) >= 2:
        pt1 = (int(intersections[0][0]), int(intersections[0][1]))
        pt2 = (int(intersections[1][0]), int(intersections[1][1]))
        
        # Рисуем линию
        cv2.line(output_image, pt1, pt2, (0, 0, 255), 2)
        
        # Вычисляем длину
        line_length = np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)
        
        # Информация
        cv2.putText(output_image, f"Angle: {angle_deg}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output_image, f"Length: {line_length:.2f} px", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return output_image, line_length
    else:
        # Диагностика: рисуем вспомогательные элементы
        angle_rad = np.deg2rad(angle_deg)
        length = 100
        pt1 = (int(center_int[0] - length * np.cos(angle_rad)), 
               int(center_int[1] - length * np.sin(angle_rad)))
        pt2 = (int(center_int[0] + length * np.cos(angle_rad)), 
               int(center_int[1] + length * np.sin(angle_rad)))
        cv2.line(output_image, pt1, pt2, (255, 0, 255), 1)  # Фиолетовая линия
        
        # Сообщение об ошибке
        cv2.putText(output_image, "No intersections found!", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return output_image, 0

def compare_slices(dicom_files, slice_numbers, angle_deg=0):
    """Сравнивает указанные срезы с линиями под фиксированным углом"""
    if len(slice_numbers) < 2:
        print("Нужно указать минимум 2 номера срезов для сравнения")
        return
    
    results = []
    images = []
    
    for slice_num in slice_numbers:
        if slice_num < 0 or slice_num >= len(dicom_files):
            print(f"Некорректный номер среза: {slice_num}")
            continue
            
        image, contours = process_dicom_file(dicom_files[slice_num], slice_num)
        
        if not contours:
            print(f"Не удалось найти контуры на срезе {slice_num}")
            continue
            
        main_contour = max(contours, key=lambda c: c.area)
        img_with_line, length = draw_contour_with_fixed_angle_line(image, main_contour, angle_deg)
        
        results.append((slice_num, length))
        images.append(img_with_line)
    
    if len(results) < 2:
        print("Недостаточно данных для сравнения")
        return
    
    # Отображение результатов
    plt.figure(figsize=(6 * len(results), 6))
    plt.suptitle(f"Сравнение срезов (линии под {angle_deg}°)", fontsize=14)
    
    for i, (img, (slice_num, length)) in enumerate(zip(images, results)):
        plt.subplot(1, len(results), i+1)
        plt.imshow(img)
        plt.title(f"Срез {slice_num}\nДлина: {length:.2f} px")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Сравнение результатов
    print("\nРезультаты сравнения:")
    for i in range(len(results)-1):
        len1 = results[i][1]
        len2 = results[i+1][1]
        diff = abs(len1 - len2)
        ratio = diff / min(len1, len2) * 100 if min(len1, len2) > 0 else 0
        
        print(f"Срез {results[i][0]} vs {results[i+1][0]}:")
        print(f"{len1:.2f} px vs {len2:.2f} px, разница: {diff:.2f} px ({ratio:.1f}%)")

def main():
    """Основная функция с возможностью выбора срезов и угла"""
    dicom_dir = input("Введите путь к папке с DICOM-файлами: ").strip()
    
    if not os.path.exists(dicom_dir):
        print(f"Ошибка: папка '{dicom_dir}' не существует!")
        return
    
    # Сбор DICOM-файлов
    dicom_files = []
    for f in os.listdir(dicom_dir):
        file_path = os.path.join(dicom_dir, f)
        if os.path.isfile(file_path):
            try:
                pydicom.dcmread(file_path, stop_before_pixels=True)
                dicom_files.append(file_path)
            except:
                continue
    
    if not dicom_files:
        print("Не найдено DICOM-файлов в указанной папке")
        return
    
    dicom_files.sort()
    print(f"\nНайдено {len(dicom_files)} DICOM-файлов (номера срезов: 0-{len(dicom_files)-1})")
    
    # Запрос номеров срезов для сравнения
    slices_input = input("Введите номера срезов для сравнения (через пробел, например '0 50 107'): ").strip()
    try:
        slice_numbers = list(map(int, slices_input.split()))
    except:
        print("Ошибка ввода номеров срезов")
        return
    
    # Запрос угла для линий
    angle = float(input("Введите угол для линий (в градусах, 0-горизонталь): ") or 0)
    
    # Сравнение срезов
    compare_slices(dicom_files, slice_numbers, angle)

if __name__ == "__main__":
    main()