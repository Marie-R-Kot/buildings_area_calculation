"""Модуль отвечает за взаимодействие с моделью сегментации, а так же сопутствующие задачи
для инференса – в том числе подсчёт площади застройки"""

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch

from PIL import Image

from utils.image_operations import ImageOperations


def load_checkpoint(path: str, model: torch.nn.Module) -> torch.nn.Module:
    """Загрузка чекпоинта модели"""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def load_model(model_path: str, encoder_name="efficientnet-b2") -> torch.nn.Module:
    """Загрузка модели"""

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    ).to("cpu")

    model = load_checkpoint(model_path, model)
    model.eval()

    return model


def get_masks(
    model: torch.nn.Module,
    tiles: list,
    positions: list,
    device="cpu",
) -> tuple:
    """Инференс модели, получаем маски и их расположение"""
    batch_size = 8
    all_masks = []
    all_positions = []

    for i in range(0, len(tiles), batch_size):
        batch = tiles[i : i + batch_size].to(device)
        batch_positions = positions[i : i + batch_size]

        with torch.no_grad():
            output = model(batch)
            prob = torch.sigmoid(output)
            mask = (prob > 0.5).float()

        for j in range(mask.shape[0]):
            mask_np = mask[j].squeeze().cpu().numpy().astype(np.uint8) * 255

            mask_pil = Image.fromarray(mask_np, mode="L")
            all_masks.append(mask_pil)
            all_positions.append(batch_positions[j])

    if not all_masks:
        return None, None

    return all_masks, all_positions


def mosaic_image(masks: list, positions: list, or_size: int) -> Image.Image:
    """Собираем все маски в одну, согласно сохраненным положения"""
    tile_size = masks[0].size[0]

    all_x = [pos[0] for pos in positions]
    all_y = [pos[1] for pos in positions]

    min_x, min_y = min(all_x), min(all_y)

    cols = [(x - min_x) // tile_size for x in all_x]
    rows = [(y - min_y) // tile_size for y in all_y]

    grid_cols = max(cols) + 1
    grid_rows = max(rows) + 1

    mosaic_width = grid_cols * tile_size
    mosaic_height = grid_rows * tile_size

    mosaic = Image.new("L", (mosaic_width, mosaic_height), color=0)

    for mask_pil, col, row in zip(masks, cols, rows, strict=False):
        x = col * tile_size
        y = row * tile_size
        mosaic.paste(mask_pil, (x, y))

    # Для оригинального изображения изменялся размер перед подачей модели, соответственно
    # маска была так же измененного размера, возвращаем изначальный, чтоб не было
    # конфликта в интерфейсе и не нужно было пересчитывать разрешение
    resized_image = cv2.resize(
        np.array(mosaic), (or_size, or_size), interpolation=cv2.INTER_NEAREST
    )
    pil_image = Image.fromarray(resized_image)

    return pil_image


def predict_image(
    model_path: str,
    image_pil: Image.Image,
    real_resolution: float,
) -> tuple:
    """
    Головная функция: 
    * Загружает модель
    * Отдает данные на инференс,
    * Подсчитывает площадь застройки
    """
    model = load_model(model_path)
    image_cl = ImageOperations(image_pil, inference=True)
    tiles, positions = image_cl.prepare_image_for_inference()

    all_masks, all_positions = get_masks(model, tiles, positions)

    image = mosaic_image(all_masks, all_positions, image_pil.size[0])

    mask_np = np.array(image)  # ← полная обрезанная маска
    built_up_pixels = np.sum(mask_np > 127)
    total_pixels = mask_np.size

    built_up_area_m2 = built_up_pixels * (real_resolution ** 2)
    percent = built_up_pixels / total_pixels * 100

    return image, round(built_up_area_m2), round(percent)
