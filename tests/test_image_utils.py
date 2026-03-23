#!/usr/bin/env python3

import os

from unittest import TestCase

from io import BytesIO

from image_utils import img_to_base64
from image_utils import img_to_bytesio
from image_utils import base64_to_img

from media_utils import get_image_file_info

from PIL import Image


class TestImageUtils(TestCase):

    def test_base64(self) -> None:
        image = Image.new('RGB', (64, 48), color='red')

        base64_str = img_to_base64(image)
        self.assertIsInstance(base64_str, str)
        assert base64_str is not None

        image_from_base64 = base64_to_img(base64_str)
        self.assertIsInstance(image_from_base64, Image.Image)
        self.assertEqual(image.size, image_from_base64.size)

        base64_str = img_to_base64(None)
        self.assertIsNone(base64_str)

        image_bytesio = img_to_bytesio(image)
        self.assertIsInstance(image_bytesio, BytesIO)
        image_bytesio = img_to_bytesio(None)
        self.assertIsNone(image_bytesio)

        image_path = "test_image.png"
        with open(image_path, "wb") as f:
            image.save(f, format="PNG")
        image_info = get_image_file_info(image_path)
        self.assertEqual(image_info['width'], 64)
        self.assertEqual(image_info['height'], 48)
        self.assertAlmostEqual(image_info['aspect_ratio'], 4.0 / 3.0, delta=0.1)

        with self.assertRaises(TypeError):
            img_to_base64(12345)
        with self.assertRaises(TypeError):
            base64_to_img(12345)  # type: ignore[arg-type]
        with self.assertRaises(Exception):
            base64_to_img("not-a-real-base64-string")
        with self.assertRaises(TypeError):
            img_to_bytesio(12345)
        with self.assertRaises(FileNotFoundError):
            get_image_file_info("non_existent_file.png")
        with self.assertRaises(TypeError):
            get_image_file_info(["list", "of", "files"])  # type: ignore[arg-type]

        os.remove(image_path)
        del image
        del base64_str
        del image_from_base64
