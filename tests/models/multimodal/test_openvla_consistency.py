# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for OpenVLA model verifying vLLM output against reference tokens.

OpenVLA is a Vision-Language-Action model that outputs 7 discretized action tokens
(xyz, rpy, gripper) with 256 bins each (token IDs 31744-31999 in Llama vocab).

Reference tokens were captured from vLLM on gfx1151 (ROCm, bfloat16,
enforce_eager=True) with a deterministic seed-42 test image and temperature=0.
Tokens may differ on other platforms due to numerical precision; update
REFERENCE_TOKENS if running on a different GPU architecture or dtype.

Usage:
    pytest tests/models/multimodal/test_openvla_consistency.py -v
    # Or as standalone script:
    python tests/models/multimodal/test_openvla_consistency.py
"""

import numpy as np
import pytest
from PIL import Image

from vllm import LLM, SamplingParams

MODEL_ID = "openvla/openvla-7b"

ACTION_TOKEN_MIN = 31744
ACTION_TOKEN_MAX = 31999

# Reference tokens captured from vLLM (gfx1151, bfloat16, seed=42, temp=0).
REFERENCE_TOKENS: dict[str, list[int]] = {
    "pick up the red block": [31884, 31824, 31872, 31808, 31815, 31768, 31872],
    "move the cube to the left": [31867, 31843, 31951, 31842, 31810, 31880, 31872],
    "push the ball forward": [31782, 31810, 31999, 31893, 31856, 31827, 31744],
    "place the object on the table": [31873, 31829, 31887, 31862, 31839, 31851, 31744],
    "grasp the yellow cylinder": [31884, 31871, 31917, 31882, 31850, 31848, 31744],
}

INSTRUCTIONS = list(REFERENCE_TOKENS.keys())


def create_test_image(seed: int = 42) -> Image.Image:
    """Create a deterministic test image for reproducibility."""
    rng = np.random.default_rng(seed)
    # Create a simple 224x224 RGB image with some structure
    img_array = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


def format_prompt(instruction: str) -> str:
    """Format instruction into OpenVLA prompt format.

    Note: Trailing space is required for HF compatibility.
    """
    return f"<PAD>In: What action should the robot take to {instruction}?\nOut: "


def get_vllm_tokens(
    llm: LLM,
    instruction: str,
    image: Image.Image,
) -> list[int]:
    """Get action tokens from vLLM inference.

    Returns:
        List of 7 action token IDs from the model output.

    """
    prompt = format_prompt(instruction)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=7,
    )

    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        },
        sampling_params=sampling_params,
    )

    # Extract token IDs from output
    action_tokens = list(outputs[0].outputs[0].token_ids)
    return action_tokens


@pytest.fixture(scope="module")
def vllm_model():
    """Initialize vLLM model once for all tests."""
    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=512,
        enforce_eager=True,
    )
    yield llm


@pytest.fixture(scope="module")
def test_image():
    """Create test image once for all tests."""
    return create_test_image(seed=42)


@pytest.mark.slow
@pytest.mark.parametrize("instruction", INSTRUCTIONS)
def test_openvla_token_regression(
    vllm_model,
    test_image,
    instruction: str,
):
    """Test that vLLM produces expected action tokens for each instruction."""
    vllm_tokens = get_vllm_tokens(vllm_model, instruction, test_image)
    ref_tokens = REFERENCE_TOKENS[instruction]

    assert len(vllm_tokens) == 7, f"Expected 7 tokens, got {len(vllm_tokens)}"
    for token in vllm_tokens:
        assert ACTION_TOKEN_MIN <= token <= ACTION_TOKEN_MAX, (
            f"Token {token} outside action range "
            f"[{ACTION_TOKEN_MIN}, {ACTION_TOKEN_MAX}]"
        )

    assert vllm_tokens == ref_tokens, (
        f"Token mismatch for '{instruction}':\n"
        f"  vLLM: {vllm_tokens}\n"
        f"  Ref:  {ref_tokens}"
    )


if __name__ == "__main__":
    # Run as standalone script for quick verification.
    print("OpenVLA Regression Test")
    print("=" * 60)

    image = create_test_image(seed=42)
    print(f"Created test image: {image.size}")

    print("\nInitializing vLLM...")
    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=512,
        enforce_eager=True,
    )

    print("\nComparing vLLM outputs to reference tokens:")
    print("-" * 60)

    exact_matches = 0
    for i, instruction in enumerate(INSTRUCTIONS):
        vllm_tokens = get_vllm_tokens(llm, instruction, image)
        ref_tokens = REFERENCE_TOKENS[instruction]

        is_exact = vllm_tokens == ref_tokens

        if is_exact:
            exact_matches += 1
            status = "EXACT MATCH"
        else:
            matches = sum(1 for v, r in zip(vllm_tokens, ref_tokens) if v == r)
            status = f"DIFF ({matches}/7 tokens)"

        symbol = "OK" if is_exact else "FAIL"
        print(f"Sample {i}: {symbol} {status}")
        print(f"  Instruction: {instruction}")
        print(f"  vLLM: {vllm_tokens}")
        print(f"  Ref:  {ref_tokens}")

        in_range = all(ACTION_TOKEN_MIN <= t <= ACTION_TOKEN_MAX for t in vllm_tokens)
        if not in_range:
            print(f"  WARNING: tokens outside [{ACTION_TOKEN_MIN}, {ACTION_TOKEN_MAX}]")
        print()

    print("=" * 60)
    print(
        f"Results: {exact_matches}/{len(INSTRUCTIONS)} exact matches "
        f"({exact_matches / len(INSTRUCTIONS):.0%})"
    )

    if exact_matches == len(INSTRUCTIONS):
        print("PASS: All samples match reference tokens")
    else:
        print("FAIL: Some samples diverged from reference")
