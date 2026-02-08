"""Модуль осуществляет работу с картами через различные API"""

import io
import math

from concurrent.futures import ThreadPoolExecutor

import requests

from PIL import Image


def get_coords_by_address(address: str) -> tuple[float, float]:
    """
    Получает координаты по адресу через OpenStreetMap
    """
    base_url = "https://nominatim.openstreetmap.org/search"

    params = {"q": address, "format": "json", "limit": 1, "accept-language": "ru"}

    headers = {"User-Agent": "StreamlitApp/1.0"}

    response = requests.get(base_url, params=params, headers=headers, timeout=10)
    response.raise_for_status()

    data = response.json()

    first_result = data[0]
    lat = float(first_result["lat"])
    lon = float(first_result["lon"])

    return lat, lon


def calculate_optimal_zoom(
    lat: float, target_resolution: float = 0.3
) -> tuple[int, float]:
    """
    Вычисляет оптимальный зум для заданного разрешения (м/пиксель).

    @target_resolution: целевое разрешение в метрах на пиксель
    """
    # Формула для Web Mercator (используется ESRI API)
    # На экваторе: resolution = 156543.03392 / (2^zoom)
    # С учетом широты: resolution = 156543.03392 * cos(latitude) / (2^zoom)
    
    equator_length = 156543.03392
    lat_rad = math.radians(lat)
    real_resolution = 1

    best_zoom = 18
    best_diff = float("inf")

    for zoom in range(18, 1, -1):
        resolution = equator_length * math.cos(lat_rad) / (2 ** zoom)

        diff = abs(resolution - target_resolution)
        if diff < best_diff:
            best_diff = diff
            best_zoom = zoom
            real_resolution = resolution
        else:
            continue

    return best_zoom, real_resolution


def get_tile_bounds(xtile: int, ytile: int, zoom: int) -> tuple:
    """Возвращает границы тайла в градусах (север, юг, запад, восток)"""
    lat_north, lon_west = num2deg(xtile, ytile, zoom)
    lat_south, lon_east = num2deg(xtile + 1, ytile + 1, zoom)

    return lat_north, lat_south, lon_west, lon_east


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple:
    """Конвертация координат в номер тайла"""
    lat_rad = math.radians(lat_deg)
    n = 2.0**zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def num2deg(xtile: int, ytile: int, zoom: int) -> tuple:
    """Конвертация номера тайла в координаты"""
    n = 2.0**zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def download_tile(coords: tuple) -> Image.Image:
    """Загрузка одного тайла.
    Всегда возвращает изображение (или серую заглушку при ошибке)
    """
    zoom, x, y = coords
    url = (
        "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
        f"MapServer/tile/{zoom}/{y}/{x}"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception:
        return Image.new("RGB", (256, 256), color="gray")


def get_tiles(tile_coords_list: list) -> dict[tuple, Image.Image]:
    """
    Загрузка тайлов параллельно.
    Возвращает словарь {(x, y): tile}
    """
    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        # map автоматически сопоставит результаты с исходными координатами
        tiles = executor.map(download_tile, tile_coords_list, timeout=30)
        for coord, tile in zip(tile_coords_list, tiles, strict=False):
            # tile_coords_list состоит из: [(zoom, x, y)]
            # заменяем zoom на _, чтобы не передавать лишнее, т.к. zoom один и тот же
            _, x, y = coord
            results[(x, y)] = tile
    return results


def latlon_to_pixel(
    lat: float,
    lon: float,
    zoom: int,
    origin_x: int,
    origin_y: int,
) -> tuple:
    """Конвертирует координаты в пиксели относительно мозаики"""
    # Получаем глобальные пиксельные координаты
    lat_rad = math.radians(lat)
    n = 2.0**zoom
    global_x = (lon + 180.0) / 360.0 * n * 256
    global_y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n * 256

    # Переводим к координатам мозаики
    local_x = global_x - (origin_x * 256)
    local_y = global_y - (origin_y * 256)

    return local_x, local_y


def get_bounds(folium_bounds: dict) -> dict:
    """Преобразует границы из формата folium"""
    bounds = {}
    bounds["south"] = folium_bounds["_southWest"]["lat"]
    bounds["west"] = folium_bounds["_southWest"]["lng"]
    bounds["north"] = folium_bounds["_northEast"]["lat"]
    bounds["east"] = folium_bounds["_northEast"]["lng"]

    return bounds


def get_image(bounds: dict, target_resolution=0.3) -> tuple:
    """
    Создает большое изображение области, заданной overview_zoom,
    состоящее из тайлов с разрешением target_resolution.
    """
    north, south, west, east = (
        bounds["north"],
        bounds["south"],
        bounds["west"],
        bounds["east"],
    )

    # Вычисляем центральную точку области для определения зума
    center_lat = (north + south) / 2

    detail_zoom, real_resolution = calculate_optimal_zoom(center_lat, target_resolution)

    x_min_detail, y_max_detail = deg2num(north, west, detail_zoom)
    x_max_detail, y_min_detail = deg2num(south, east, detail_zoom)

    # Для разных полушарий могут быть иначе распределены значения
    x_min_detail, x_max_detail = sorted([x_min_detail, x_max_detail])
    y_min_detail, y_max_detail = sorted([y_min_detail, y_max_detail])

    # Формируем список всех необходимых тайлов
    tile_coords = []
    for x in range(x_min_detail, x_max_detail + 1):
        for y in range(y_min_detail, y_max_detail + 1):
            tile_coords.append((detail_zoom, x, y))

    total_tiles = len(tile_coords)
    print(f"Скачано тайлов: {total_tiles}")

    if total_tiles == 0:
        raise ValueError("Нет тайлов для загрузки. Проверьте границы.")

    tiles_dict = get_tiles(tile_coords)

    if not tiles_dict:
        raise RuntimeError("Не удалось загрузить ни одного тайла.")

    min_x = min(x for x, _ in tiles_dict.keys())
    min_y = min(y for _, y in tiles_dict.keys())

    mosaic_width = (x_max_detail - x_min_detail + 1) * 256
    mosaic_height = (y_max_detail - y_min_detail + 1) * 256

    mosaic = Image.new("RGB", (mosaic_width, mosaic_height))

    for (tile_x, tile_y), tile in tiles_dict.items():
        pos_x = (tile_x - min_x) * 256
        pos_y = (tile_y - min_y) * 256
        mosaic.paste(tile, (pos_x, pos_y))

    # Точная обрезка по географическим границам
    # Вычисляем позицию углов границ внутри мозаики в пикселях
    # Скорее всего тайлы esri перекрывают необходимую область с запасом,
    # поэтому требуется обрезка

    # Координаты углов мозаики в глобальных тайлах
    origin_x = x_min_detail
    origin_y = y_min_detail

    # Вычисляем границы обрезки
    left_px, top_px = latlon_to_pixel(north, west, detail_zoom, origin_x, origin_y)
    right_px, bottom_px = latlon_to_pixel(south, east, detail_zoom, origin_x, origin_y)

    # Обрезаем (округляем до целых пикселей)
    left_px = max(0, int(left_px))
    top_px = max(0, int(top_px))
    right_px = min(mosaic_width, int(right_px))
    bottom_px = min(mosaic_height, int(bottom_px))

    if right_px <= left_px or bottom_px <= top_px:
        print("Неправильные границы обрезки, возвращаю полную мозаику")
        return mosaic

    cropped = mosaic.crop((left_px, top_px, right_px, bottom_px))

    return cropped, real_resolution
