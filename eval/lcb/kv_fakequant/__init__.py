# SPDX-License-Identifier: Apache-2.0
"""Cache-level KV fake-quantization for the LiveCodeBench KV-quant comparison.

    build_quantizer(method, **kw) -> KVQuantizer | None
    FakeQuantKVCache(quantizer, config, sink=, residual_length=) -> transformers Cache
    available_methods() -> list[str]
"""

from .quantizers import (build_quantizer, available_methods, KVQuantizer, PUBLISHED_RECIPE)
from .cache import FakeQuantKVCache

__all__ = ["build_quantizer", "available_methods", "KVQuantizer", "FakeQuantKVCache",
           "PUBLISHED_RECIPE"]
