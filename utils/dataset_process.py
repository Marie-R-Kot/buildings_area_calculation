"""Модуль содержит классы для обработки изображений датасета и отдельных картинок
перед обучением или инференсом"""

import os

from tqdm import tqdm

from utils.image_operations import ImageOperations


class DataOperations:
    """Класс содержит методы обработки датасета"""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        files_dist: dict,
        save_resized: bool = None,
        blank_threshold: int = 600,
    ) -> None:
        self.images_dir = os.path.join(input_dir, "images")
        self.masks_dir = os.path.join(input_dir, "gt")
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)
        for type in ["train", "val", "test"]:
            os.makedirs(f"{output_dir}/{type}/images", exist_ok=True)
            os.makedirs(f"{output_dir}/{type}/gt", exist_ok=True)
        os.makedirs(f"{output_dir}/images_w_mask", exist_ok=True)

        self.train_dir = os.path.join(output_dir, "train")
        self.val_dir = os.path.join(output_dir, "val")
        self.test_dir = os.path.join(output_dir, "test")

        self.out_resized = None
        if save_resized:
            self.out_resized = os.path.join(output_dir, "resized")

        self.files_distribution = files_dist
        self.blank_threshold = blank_threshold

    def _check_file(self, file_name: str) -> str:
        """Метод возвращает тип выборки, к которой относится данный файл"""
        for key, value in self.files_distribution.items():
            if file_name in value:
                return key

    def process_data(
        self,
        resize: int = 5120,
        tile_size: int = 512,
        overlap: int = 0,
        save_resized: bool = False,
    ) -> None:
        """
        Метод осуществялет предобработку датасета, сохраняет данные после resize
        и разрезания на тайлы в директории соответствующие подвыборкам
        - main filder
            - train
                - images
                - gt
            - val
                ...
            - test
             ...
        ! Не возвращает преобразованные изображения
        """
        if save_resized:
            self.out_resized = os.path.join(self.output_dir, "resized")

        image_files = sorted(
            [f for f in os.listdir(self.images_dir) if f.endswith(".tif")]
        )

        blank_counts = 0

        for img_file in tqdm(image_files, desc="Processing files"):
            file_type = self._check_file(img_file)

            image_cl = ImageOperations(img_file, self.images_dir, self.masks_dir)

            resized_image, resized_mask = image_cl.resize_image(
                image_cl.image,
                image_cl.mask,
                target_size=resize,
                output_dir=self.out_resized,
            )

            _, blank_counts = image_cl.cut_image_to_tiles(
                resized_image,
                resized_mask,
                file_type,
                self.output_dir,
                blank_counts,
                tile_size,
                overlap,
                self.blank_threshold,
            )
