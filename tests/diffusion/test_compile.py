# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch.nn as nn

import vllm_omni.diffusion.compile as compile_module
from vllm_omni.diffusion.compile import regionally_compile

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _WrappedBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.compile_called = False
        self.forward_compiled = False

    def compile(self, *args, **kwargs):
        self.compile_called = True
        return self

    def forward(self, x):
        return x


class _ModelWithWrappedRepeatedBlocks(nn.Module):
    _repeated_blocks = ["OriginalBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks"]

    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_WrappedBlock(), _WrappedBlock()])
        self.other_blocks = nn.ModuleList([_WrappedBlock()])


def test_regionally_compile_matches_wrapped_blocks_by_declared_container_attr(monkeypatch):
    model = _ModelWithWrappedRepeatedBlocks()
    compile_calls = []

    def _compile(fn, *args, **kwargs):
        compile_calls.append((fn, args, kwargs))

        def _compiled(*fn_args, **fn_kwargs):
            return f"compiled:{fn(*fn_args, **fn_kwargs)}"

        return _compiled

    monkeypatch.setattr(compile_module.torch, "compile", _compile)

    regionally_compile(model, dynamic=True)

    assert len(compile_calls) == 2
    assert all(not block.compile_called for block in model.transformer_blocks)
    assert not model.other_blocks[0].compile_called
    assert model.transformer_blocks[0].forward("ok") == "compiled:ok"


def test_regionally_compile_does_not_partially_mutate_on_setup_failure(monkeypatch):
    model = _ModelWithWrappedRepeatedBlocks()
    original_forwards = [block.forward.__func__ for block in model.transformer_blocks]
    compile_calls = 0

    def _compile(fn, *args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        if compile_calls == 2:
            raise RuntimeError("compile setup failed")
        return lambda *fn_args, **fn_kwargs: fn(*fn_args, **fn_kwargs)

    monkeypatch.setattr(compile_module.torch, "compile", _compile)

    with pytest.raises(RuntimeError, match="compile setup failed"):
        regionally_compile(model, dynamic=True)

    assert [block.forward.__func__ for block in model.transformer_blocks] == original_forwards


def test_compiled_block_falls_back_to_eager_on_inductor_error(monkeypatch):
    """A lazy compiler failure must not take the engine down.

    torch.compile work happens on the first call, so an inductor bug (e.g.
    CantSplit on fp8 FLUX with two dynamic dims, build 2953 Quantization
    Test) surfaces at dummy-run time. The block must revert to eager and
    stay eager.
    """
    from torch._inductor.exc import InductorError

    model = _ModelWithWrappedRepeatedBlocks()
    attempts = []

    def _compile(fn, *args, **kwargs):
        def _compiled(*fn_args, **fn_kwargs):
            attempts.append("compiled")
            raise InductorError(RuntimeError("CantSplit: 15360*s31 + 15360*s87"), None)

        return _compiled

    monkeypatch.setattr(compile_module.torch, "compile", _compile)
    regionally_compile(model)

    block = model.transformer_blocks[0]
    assert block.forward("x") == "x"  # first call: compiled raises, eager answers
    assert block.forward("y") == "y"  # second call: stays eager
    assert attempts == ["compiled"]  # compiled path was not retried


def test_compiled_block_does_not_swallow_model_errors(monkeypatch):
    """Real model failures inside the compiled forward must propagate."""
    model = _ModelWithWrappedRepeatedBlocks()

    def _compile(fn, *args, **kwargs):
        def _compiled(*fn_args, **fn_kwargs):
            raise ValueError("genuine model bug")

        return _compiled

    monkeypatch.setattr(compile_module.torch, "compile", _compile)
    regionally_compile(model)

    with pytest.raises(ValueError, match="genuine model bug"):
        model.transformer_blocks[0].forward("x")
