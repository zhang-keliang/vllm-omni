# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for NumPy support in the image-output coercion helpers.

LingBot T2I fills ``OmniRequestOutput.images`` with NumPy arrays (its
postprocess converts tensors to NumPy for the image serving path). The
text-to-image doc example then crashed on ``images[0].save(...)`` with
``AttributeError: 'numpy.ndarray' object has no attribute 'save'``
(build 2952, Diffusion X2I Doc Test). ``coerce_images_to_pil`` /
``extract_images_from_outputs`` must turn every supported payload —
PIL, torch tensors, NumPy arrays, nested lists — into PIL images.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.utils.image_output import (
    coerce_images_to_pil,
    extract_images_from_outputs,
)
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_coerce_float_ndarray_zero_one_range():
    array = np.full((8, 12, 3), 0.5, dtype=np.float32)
    images = coerce_images_to_pil(array)
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
    assert images[0].size == (12, 8)
    assert np.asarray(images[0])[0, 0, 0] == 127


def test_coerce_float_ndarray_minus_one_one_range():
    array = np.full((8, 12, 3), -1.0, dtype=np.float32)
    images = coerce_images_to_pil(array)
    assert len(images) == 1
    assert np.asarray(images[0]).max() == 0


def test_coerce_uint8_ndarray_preserves_values():
    array = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    images = coerce_images_to_pil(array)
    assert len(images) == 1
    assert np.array_equal(np.asarray(images[0]), array)


def test_coerce_batched_ndarray_yields_one_image_per_item():
    array = np.zeros((2, 8, 12, 3), dtype=np.float32)
    images = coerce_images_to_pil(array)
    assert len(images) == 2
    assert all(isinstance(img, Image.Image) for img in images)


def test_coerce_channel_first_integer_ndarray():
    array = np.zeros((3, 8, 12), dtype=np.uint8)
    images = coerce_images_to_pil(array)
    assert len(images) == 1
    assert images[0].size == (12, 8)


def test_coerce_mixed_list_passthrough_and_conversion():
    pil = Image.new("RGB", (4, 4))
    payload = [pil, np.zeros((8, 12, 3), dtype=np.float32), torch.zeros(3, 8, 12)]
    images = coerce_images_to_pil(payload)
    assert len(images) == 3
    assert images[0] is pil
    assert all(isinstance(img, Image.Image) for img in images)


def test_extract_images_from_outputs_coerces_ndarray_images():
    # The exact LingBot shape: OmniRequestOutput.images holding a NumPy array.
    output = OmniRequestOutput(
        request_id="lingbot-doc-example",
        images=[np.zeros((8, 12, 3), dtype=np.float32)],
    )
    images = extract_images_from_outputs(output)
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
