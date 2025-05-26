import os
import numpy as np
import pydicom
import vtk
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import vtkMarchingCubes
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor, vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import logging
import argparse

# Настройка логирования
logging.basicConfig(level=logging.INFO)

def load_dicom_series(directory):
    """
    Загружает серию DICOM-файлов из указанной директории.
    Фильтрует срезы по наиболее распространенному размеру и сортирует их по положению.
    """
    try:
        dicom_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".dcm")]
        dicom_files.sort()
        
        slices = []
        size_counts = {}
        for f in dicom_files:
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True)
                size = (ds.Rows, ds.Columns)
                size_counts[size] = size_counts.get(size, 0) + 1
                slices.append(ds)
            except Exception as e:
                logging.error(f"Error reading {f}: {e}")
        
        most_common_size = max(size_counts, key=size_counts.get)
        logging.info(f"Most common size: {most_common_size}, Count: {size_counts[most_common_size]}")

        filtered_slices = [s for s in slices if (s.Rows, s.Columns) == most_common_size]
        if not filtered_slices:
            raise ValueError("No DICOM images found with the most common size.")

        filtered_slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        images = np.stack([pydicom.dcmread(s.filename).pixel_array for s in filtered_slices], axis=-1)
        return images, filtered_slices
    except Exception as e:
        logging.error(f"Error loading DICOM series: {e}")
        raise

def convert_to_vtk_image(volume):
    """
    Конвертирует 3D-массив NumPy в объект vtkImageData.
    """
    volume = np.ascontiguousarray(volume, dtype=np.uint16)  # Обеспечиваем непрерывность данных
    vtk_data = vtkImageData()
    vtk_data.SetDimensions(volume.shape)
    vtk_data.AllocateScalars(vtk.VTK_UNSIGNED_SHORT, 1)

    flat_data = volume.ravel(order="F")  
    vtk_array = vtk_data.GetPointData().GetScalars()
    vtk_array.SetVoidArray(flat_data, len(flat_data), 1)
    return vtk_data

def create_3d_model(volume, isovalue=200, cut_y_plane=None, cut_point=None, cut_normal=(1, 0, 0), background_color=(0.1, 0.1, 0.1), model_color=(1, 1, 1)):
    """
    Создает 3D-модель из объема данных с использованием Marching Cubes.
    Возможна обрезка модели по плоскости и настройка визуализации.

    Параметры:
    - volume: 3D-массив данных.
    - isovalue: Значение изоповерхности для Marching Cubes.
    - cut_y_plane: Если задано, обрезает модель по плоскости Y.
    - cut_point: Если задано, обрезает модель по плоскости, проходящей через эту точку.
    - cut_normal: Нормаль плоскости для обрезки по точке (по умолчанию (1, 0, 0)).
    - background_color: Цвет фона.
    - model_color: Цвет модели.
    """
    vtk_volume = convert_to_vtk_image(volume)

    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(vtk_volume)
    mc.ComputeNormalsOn()
    mc.SetValue(0, isovalue)  
    mc.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(mc.GetOutputPort())
    mapper.ScalarVisibilityOff()  

    # Обрезка по плоскости Y, если задано
    if cut_y_plane is not None:
        clip_plane = vtk.vtkPlane()
        clip_plane.SetOrigin(0, cut_y_plane, 0)
        clip_plane.SetNormal(0, -1, 0)
        
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputConnection(mc.GetOutputPort())
        clipper.SetClipFunction(clip_plane)
        clipper.Update()

        mapper.SetInputConnection(clipper.GetOutputPort())

    # Обрезка по точке, если задано
    if cut_point is not None:
        clip_plane = vtk.vtkPlane()
        clip_plane.SetOrigin(cut_point)  # Точка, через которую проходит плоскость
        clip_plane.SetNormal(cut_normal)  # Нормаль плоскости
        
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputConnection(mc.GetOutputPort())
        clipper.SetClipFunction(clip_plane)
        clipper.Update()

        mapper.SetInputConnection(clipper.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(model_color)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(*background_color)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    render_window.Render()
    interactor.Start()

    # Сохранение модели в формате STL
    writer = vtk.vtkSTLWriter()
    writer.SetFileName("output_model.stl")
    writer.SetInputConnection(mc.GetOutputPort())
    writer.Write()

def preprocess_volume(volume, sigma=1.2):
    """
    Нормализует и сглаживает объем данных.
    """
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume)) * 255
    volume = volume.astype(np.uint16)
    volume = gaussian_filter(volume, sigma=sigma)
    return volume

def main(dicom_directory, isovalue=80, sigma=1.2):
    """
    Основная функция для загрузки, обработки и визуализации DICOM-данных.
    """
    # Загрузка серии с фильтрацией по размеру
    volume, slices = load_dicom_series(dicom_directory)

    print(f"Volume shape: {volume.shape}")
    print(f"Min value: {np.min(volume)}, Max value: {np.max(volume)}")

    # Визуализация первых пяти срезов
    num_slices = volume.shape[2]
    plt.figure(figsize=(15, 5))
    for i in range(5):
        index = i * (num_slices // 5)
        plt.subplot(1, 5, i + 1)
        plt.imshow(volume[:, :, index], cmap="gray")
        plt.title(f"Slice {index}")
        plt.axis("off")
    plt.savefig("slices.png")
    
    # Нормализация и сглаживание
    volume = preprocess_volume(volume, sigma=sigma)
    trimmed_volume = volume[:, :, :-25]

    # Построение 3D модели
    create_3d_model(trimmed_volume, isovalue=isovalue, cut_y_plane=volume.shape[1] // 2, cut_point=(127.5, 127.5, 127.5), cut_normal=(0, 1, 0))

if __name__ == "__main__":
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Process and visualize DICOM series.")
    parser.add_argument("dicom_directory", type=str, help="Path to the directory containing DICOM files")
    parser.add_argument("--isovalue", type=int, default=80, help="Isovalue for 3D model generation")
    parser.add_argument("--sigma", type=float, default=1.2, help="Sigma for Gaussian filter")
    args = parser.parse_args()

    # Запуск основной функции
    main(args.dicom_directory, args.isovalue, args.sigma)