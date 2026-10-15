# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pickle
from array import array

import pytest
import torch

from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.multimodal.utils import strip_covered_mm_data
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.request import Request


def _create_new_requests_data(prompt_embeds: torch.Tensor | None) -> NewRequestData:
    return NewRequestData(
        req_id="test_req",
        prompt_token_ids=None,
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([],),
        num_computed_tokens=0,
        lora_request=None,
        prompt_embeds=prompt_embeds,
    )


@pytest.mark.parametrize("protocol", [4, pickle.HIGHEST_PROTOCOL])
def test_packed_prompt_roundtrip_preserves_full_prefix(protocol: int) -> None:
    prompt = [-1, 0, 154879, 2**31 - 1] * 1024
    data = _create_new_requests_data(None)
    data.prompt_token_ids = prompt
    data.num_computed_tokens = len(prompt) - 256
    data.pack_prompt_token_ids = True

    received = pickle.loads(pickle.dumps(data, protocol=protocol))

    assert isinstance(received.prompt_token_ids, array)
    assert received.prompt_token_ids.itemsize == 4
    assert received.prompt_token_ids.tolist() == prompt
    assert received.prompt_len == len(prompt)
    assert received.num_computed_tokens == len(prompt) - 256
    assert not received.pack_prompt_token_ids
    assert data.prompt_token_ids is prompt and data.pack_prompt_token_ids

    # Mutating sender or receiver must not affect the other side.
    prompt[0] = 7
    assert received.prompt_token_ids[0] == -1
    received.prompt_token_ids[1] = 9
    assert prompt[1] == 0


@pytest.mark.parametrize("invalid", [-(2**31) - 1, 2**31])
def test_packed_prompt_rejects_overflow(invalid: int) -> None:
    data = _create_new_requests_data(None)
    data.prompt_token_ids = [invalid] + [0] * 4095
    data.pack_prompt_token_ids = True
    with pytest.raises(OverflowError):
        pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)


@pytest.mark.parametrize("prompt", [None, [], [1, 2], list(range(4095))])
def test_short_prompt_keeps_list_representation(prompt: list[int] | None) -> None:
    data = _create_new_requests_data(None)
    data.prompt_token_ids = prompt
    data.pack_prompt_token_ids = True
    received = pickle.loads(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
    assert received.prompt_token_ids == prompt
    assert received.prompt_token_ids is None or isinstance(
        received.prompt_token_ids, list
    )


def test_unpacked_long_prompt_keeps_list_representation() -> None:
    data = _create_new_requests_data(None)
    data.prompt_token_ids = list(range(4096))
    received = pickle.loads(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
    assert isinstance(received.prompt_token_ids, list)
    assert received.prompt_token_ids == data.prompt_token_ids


@pytest.mark.parametrize(
    "feature", ["text", "mrope", "multimodal", "embeds", "pooling"]
)
def test_prompt_packing_feature_guards(feature: str) -> None:
    prompt = list(range(4096))
    req = Request(
        "packing-test",
        prompt,
        None if feature == "pooling" else SamplingParams(max_tokens=1),
        PoolingParams(task="embed") if feature == "pooling" else None,
        prompt_embeds=torch.zeros(4096, 1) if feature == "embeds" else None,
        mm_features=[_mm_feature(0, 1)] if feature == "multimodal" else None,
    )
    data = NewRequestData.from_request(
        req,
        ([1],),
        uses_mrope=feature == "mrope",
        pack_prompt_token_ids=True,
    )
    received = pickle.loads(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
    assert isinstance(received.prompt_token_ids, array if feature == "text" else list)
    assert list(received.prompt_token_ids) == prompt
    assert req.prompt_token_ids is prompt


def test_repr_with_none() -> None:
    """Test repr when prompt_embeds is None."""
    new_requests_data = _create_new_requests_data(None)

    assert "prompt_embeds_shape=None" in repr(new_requests_data)
    assert "prompt_embeds_shape=None" in new_requests_data.anon_repr()


def test_repr_with_multi_element_tensor() -> None:
    """Test repr when prompt_embeds is a multi-element tensor."""
    prompt_embeds = torch.randn(10, 768)
    new_requests_data = _create_new_requests_data(prompt_embeds)

    assert "prompt_embeds_shape=torch.Size([10, 768])" in repr(new_requests_data)
    assert "prompt_embeds_shape=torch.Size([10, 768])" in new_requests_data.anon_repr()


def _mm_feature(offset: int, length: int) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=MultiModalKwargsItem.dummy(),
        mm_position=PlaceholderRange(offset=offset, length=length),
        identifier=f"hash_{offset}",
        modality="image",
    )


def test_strip_covered_mm_data() -> None:
    """Items fully inside the computed prefix lose their data; items touching
    the uncomputed region keep it; already-None data stays None."""
    from dataclasses import replace

    covered = _mm_feature(offset=0, length=100)
    boundary = _mm_feature(offset=150, length=100)  # ends exactly at 250
    uncovered = _mm_feature(offset=300, length=100)
    already_none = replace(_mm_feature(offset=100, length=50), data=None)

    stripped = strip_covered_mm_data(
        [covered, boundary, uncovered, already_none], num_computed_tokens=250
    )

    assert stripped[0].data is None  # fully covered -> stripped
    assert stripped[1].data is None  # span end == computed -> covered -> stripped
    assert stripped[2].data is not None  # extends past prefix -> kept
    assert stripped[3].data is None  # was already None
    # non-data fields are preserved
    assert stripped[0].identifier == covered.identifier
    assert stripped[0].mm_position == covered.mm_position
    # original list is not mutated
    assert covered.data is not None


def test_strip_covered_mm_data_zero_computed() -> None:
    """With no prefix hit nothing is stripped."""
    features = [_mm_feature(offset=0, length=100)]
    stripped = strip_covered_mm_data(features, num_computed_tokens=0)
    assert stripped[0].data is not None


def _mm_feature_mixed(offset: int, length: int) -> MultiModalFeatureSpec:
    data = MultiModalKwargsItem(
        {
            "pixel_values": MultiModalFieldElem(
                data=torch.empty(4), field=MultiModalBatchedField()
            ),
            "image_grid_thw": MultiModalFieldElem(
                data=torch.ones(1, 3, dtype=torch.long),
                field=MultiModalBatchedField(keep_on_cpu=True),
            ),
        }
    )
    return MultiModalFeatureSpec(
        data=data,
        mm_position=PlaceholderRange(offset=offset, length=length),
        identifier=f"hash_{offset}",
        modality="image",
    )


def test_strip_covered_mm_data_mrope() -> None:
    """For M-RoPE models, covered items keep their keep_on_cpu metadata fields
    (the worker needs them to compute positions); payload fields are dropped."""
    covered = _mm_feature_mixed(offset=0, length=100)
    uncovered = _mm_feature_mixed(offset=300, length=100)

    stripped = strip_covered_mm_data(
        [covered, uncovered], num_computed_tokens=250, uses_mrope=True
    )

    assert stripped[0].data is not None
    assert list(stripped[0].data.keys()) == ["image_grid_thw"]
    assert stripped[1].data is not None
    assert set(stripped[1].data.keys()) == {"pixel_values", "image_grid_thw"}
    # original list is not mutated
    assert set(covered.data.keys()) == {"pixel_values", "image_grid_thw"}


def test_strip_covered_mm_data_shm_address_item() -> None:
    """SHM address descriptors must survive stripping even when covered: the
    worker needs the address to resolve the payload and to acknowledge the
    sender's writer reference. Stripping it crashed the SHM receiver cache on
    the second identical request (vllm-project/vllm#54994)."""
    address_item = MultiModalKwargsItem(
        {
            "address": MultiModalFieldElem(data=4096, field=MultiModalBatchedField()),
            "monotonic_id": MultiModalFieldElem(data=7, field=MultiModalBatchedField()),
        }
    )
    feature = MultiModalFeatureSpec(
        data=address_item,
        mm_position=PlaceholderRange(offset=0, length=100),
        identifier="shm_item",
        modality="image",
    )

    stripped = strip_covered_mm_data([feature], num_computed_tokens=250)

    assert stripped[0].data is address_item
