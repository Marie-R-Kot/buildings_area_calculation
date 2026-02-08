"""Модуль содержит классы для обработки изображений датасета и входящих изображений"""

import math
import os

import cv2
import numpy as np
import torch


class ImageOperations:
    """Класс содержит методы обработки одного изображения и его маски, если она есть"""

    def __init__(
        self,
        image_input: str | np.ndarray,
        image_dir: str = None,
        mask_dir: str = None,
        inference: bool = False,
    ) -> None:
        self.inference = inference
        if self.inference:
            self.image = image_input.copy()
            self.file_name = "loaded_image"
        else:
            self.image = cv2.imread(os.path.join(image_dir, image_input))
            self.mask = cv2.imread(
                os.path.join(mask_dir, image_input), cv2.IMREAD_GRAYSCALE
            )
            self.file_name = os.path.splitext(image_input)[0]

    def resize_image(
        self,
        image: np.ndarray,
        mask: np.ndarray = None,
        target_size: int = 5120,
        output_dir: str = None,
    ) -> np.ndarray | tuple:
        """Изменение размера для удобства дальнейшей резки и работы модели"""
        target_size = (target_size, target_size)
        self.resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
        if self.inference:
            return self.resized_image

        else:
            self.resized_mask = cv2.resize(
                mask, target_size, interpolation=cv2.INTER_NEAREST
            )

            if output_dir:
                cv2.imwrite(f"{output_dir}/images/{self.file_name}", self.resized_image)
                cv2.imwrite(f"{output_dir}/gt/{self.file_name}", self.resized_mask)

            return self.resized_image, self.resized_mask

    def cut_image_to_tiles(
        self,
        image: np.ndarray,
        mask: np.ndarray = None,
        type: str = None,
        output_dir: str = None,
        blank_counts: int = 0,
        tile_size: int = 512,
        overlap: int = 0,
        blank_threshold: int = 600,
    ) -> list | tuple:
        """
        Метод разрезает изображение на тайлы меньшего размера
        В случае инференса возвращает тайлы и их положение на оригинальном изображении
        """
        if not self.inference:
            img_tiles_dir = os.path.join(os.path.join(output_dir, type), "images")
            mask_tiles_dir = os.path.join(os.path.join(output_dir, type), "gt")

        height, width = image.shape[:2]
        tile_id = 0

        step = tile_size - overlap

        tiles = []

        for y in range(0, height - tile_size + 1, step):
            for x in range(0, width - tile_size + 1, step):
                img_tile = image[y : y + tile_size, x : x + tile_size, :]

                if not self.inference:
                    mask_tile = mask[y : y + tile_size, x : x + tile_size]

                    check = np.sum(mask_tile > 0) / mask_tile.size

                    img_filename = f"{self.file_name}_tile_{tile_id:04d}_y{y}_x{x}.png"
                    mask_filename = f"{self.file_name}_tile_{tile_id:04d}_y{y}_x{x}.png"

                    if type == "train" and check == 0:
                        if blank_counts > blank_threshold:
                            cv2.imwrite(
                                os.path.join(
                                    os.path.join(output_dir, "images_w_mask"),
                                    img_filename,
                                ),
                                cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR),
                            )
                            continue
                        blank_counts += 1

                    cv2.imwrite(
                        os.path.join(img_tiles_dir, img_filename),
                        cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR),
                    )
                    cv2.imwrite(os.path.join(mask_tiles_dir, mask_filename), mask_tile)
                else:
                    tiles.append((x, y, img_tile))

                tile_id += 1

        # Для случая, если все исходное изображение покрывается одним тайлом
        if not tiles:
            tiles = [image]

        if not self.inference:
            return tile_id, blank_counts
        else:
            return tiles

    def prepare_image_for_inference(self) -> tuple:
        """Метод подготавливает сырое изображение к инференсу"""
        # Вычислем ближайший размер кратный размеры тайлов, подаваемых в модель
        height = self.image.size[0]
        target_size = int(512 * math.ceil(height / 512))

        np_image = np.array(self.image)

        resized_image = self.resize_image(np_image, target_size=target_size)
        tiles = self.cut_image_to_tiles(resized_image)
        image_tensors = []
        positions = []

        for x, y, tile in tiles:
            tile_image = np.array(tile)

            if tile_image.max() > 1.0:
                tile_image = tile_image / 255.0

            tensor_image = torch.from_numpy(tile_image.transpose(2, 0, 1)).float()

            image_tensors.append(tensor_image)
            positions.append((x, y))

        images_tensor = torch.stack(image_tensors)

        return images_tensor, positions
