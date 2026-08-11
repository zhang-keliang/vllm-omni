# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger

logger = init_logger(__name__)


def _compile_error_types() -> tuple[type[BaseException], ...]:
    """Exception types that mean "the compiler failed", not "the model failed".

    torch.compile work happens lazily on the first call, so compiler bugs
    (e.g. inductor's CantSplit on fp8 FLUX with two dynamic dims) surface at
    dummy-run time, past regionally_compile's own try/except. Only these types
    are safe to fall back on: they are raised before the compiled kernel runs.
    """
    types: list[type[BaseException]] = []
    try:
        from torch._dynamo.exc import BackendCompilerFailed

        types.append(BackendCompilerFailed)
    except ImportError:
        pass
    try:
        from torch._inductor.exc import InductorError

        types.append(InductorError)
    except ImportError:
        pass
    return tuple(types)


def _with_eager_fallback(
    block_name: str,
    eager_forward: Callable[..., Any],
    compiled_forward: Callable[..., Any],
) -> Callable[..., Any]:
    """Run the compiled forward, permanently reverting to eager on compiler errors."""
    fell_back = False

    # functools.wraps sets __wrapped__ so inspect.signature() resolves to the
    # original forward. cache-dit's BlockAdapter matches blocks by inspecting
    # forward's parameter names and return annotation (build 2954, Multi-GPU
    # Layered job); a bare (*args, **kwargs) wrapper breaks that match.
    @functools.wraps(eager_forward)
    def forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal fell_back
        if fell_back:
            return eager_forward(*args, **kwargs)
        try:
            return compiled_forward(*args, **kwargs)
        except _compile_error_types() as exc:
            fell_back = True
            logger.warning_once(
                "torch.compile failed lazily for repeated block %s (%s: %s); "
                "falling back to eager for the affected block(s).",
                block_name,
                type(exc).__name__,
                exc,
            )
            return eager_forward(*args, **kwargs)

    return forward


def _matches_repeated_block(
    name: str,
    module: nn.Module,
    repeated_blocks: list[str],
    repeated_block_attrs: list[str],
) -> bool:
    class_name = module.__class__.__name__
    if class_name in repeated_blocks:
        return True

    for attr in ("_fsdp_wrapped_module", "module", "_orig_mod"):
        wrapped = getattr(module, attr, None)
        if wrapped is not None and wrapped.__class__.__name__ in repeated_blocks:
            return True

    parts = name.split(".")
    return len(parts) >= 2 and parts[-2] in repeated_block_attrs and parts[-1].isdigit()


def regionally_compile(
    model: nn.Module,
    *compile_args: Any,
    **compile_kwargs: Any,
) -> nn.Module:
    """
    Apply regional compilation to a PyTorch model.

    Args:
        model: The PyTorch model instance to compile
        *compile_args: Positional arguments forwarded to torch.compile
        **compile_kwargs: Keyword arguments forwarded to torch.compile

    Returns:
        The same model instance (modified in-place)
    """
    # Get the list of repeated blocks from the model
    repeated_blocks = getattr(model, "_repeated_blocks", None)

    if not repeated_blocks:
        logger.warning("Regional compilation skipped because the model does not define `_repeated_blocks`.")
        return model

    repeated_block_attrs = getattr(model, "_layerwise_offload_blocks_attrs", [])

    # Build all compiled callables before mutating the model. This keeps setup
    # failures atomic: callers can safely continue with the uncompiled model if
    # torch.compile raises synchronously for any repeated block.
    compiled_forwards: list[tuple[nn.Module, Any]] = []
    for name, submod in model.named_modules():
        if _matches_repeated_block(name, submod, repeated_blocks, repeated_block_attrs):
            # Compile the block compute while keeping nn.Module.__call__ hooks
            # outside the compiled graph. NOTE: anything that wraps this
            # callable must stay signature-transparent — cache-dit's
            # BlockAdapter matches blocks by inspecting forward's parameter
            # names and return annotation, which torch.compile preserves.
            compiled_forwards.append(
                (
                    submod,
                    _with_eager_fallback(
                        name,
                        submod.forward,
                        torch.compile(submod.forward, *compile_args, **compile_kwargs),
                    ),
                )
            )

    if not compiled_forwards:
        logger.warning(f"Regional compilation skipped because {repeated_blocks} classes are not found in the model.")
    else:
        for submod, compiled_forward in compiled_forwards:
            submod.forward = compiled_forward
        logger.info(
            "Regional compilation applied to %d module(s) for repeated blocks %s.",
            len(compiled_forwards),
            repeated_blocks,
        )

    return model
