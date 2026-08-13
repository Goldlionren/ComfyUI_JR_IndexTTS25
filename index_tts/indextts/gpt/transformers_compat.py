"""Narrow adapters for private Transformers APIs used by vendored generation code."""

from __future__ import annotations

from transformers.cache_utils import Cache

try:
    from transformers.generation.configuration_utils import (
        NEED_SETUP_CACHE_CLASSES_MAPPING,
        QUANT_BACKEND_CLASSES_MAPPING,
    )
except ImportError:
    # Transformers 4.56 moved cache construction into GenerationMixin and no
    # longer exposes these private registries. IndexTTS baseline uses dynamic
    # cache (cache_implementation=None); explicit static/quantized cache modes
    # remain intentionally outside the validated compatibility surface.
    NEED_SETUP_CACHE_CLASSES_MAPPING = {}
    QUANT_BACKEND_CLASSES_MAPPING = {}


try:
    from transformers.cache_utils import QuantizedCacheConfig
except ImportError:

    class QuantizedCacheConfig:  # pragma: no cover - only used by optional quantized cache
        """Fail clearly when an obsolete quantized-cache path is explicitly requested."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "This IndexTTS compatibility layer does not support the removed "
                "Transformers QuantizedCacheConfig API. Disable quantized KV cache."
            )


try:
    from transformers.generation.candidate_generator import _crop_past_key_values
except ImportError:

    def _crop_past_key_values(model, past_key_values, max_length):
        """Transformers 4.52-compatible cache crop for legacy assisted generation."""
        new_past = []
        if isinstance(past_key_values, Cache):
            past_key_values.crop(max_length)
        elif model.config.is_encoder_decoder:
            for layer in past_key_values:
                new_past.append(
                    (
                        layer[0][:, :, :max_length, :],
                        layer[1][:, :, :max_length, :],
                        layer[2],
                        layer[3],
                    )
                )
            past_key_values = tuple(new_past)
        elif "gptbigcode" in model.__class__.__name__.lower() or (
            model.config.architectures is not None
            and "gptbigcode" in model.config.architectures[0].lower()
        ):
            for index in range(len(past_key_values)):
                if model.config.multi_query:
                    past_key_values[index] = past_key_values[index][:, :max_length, :]
                else:
                    past_key_values[index] = past_key_values[index][:, :, :max_length, :]
        elif past_key_values is not None:
            for layer in past_key_values:
                if layer != ([], []):
                    new_past.append(
                        (
                            layer[0][:, :, :max_length, :],
                            layer[1][:, :, :max_length, :],
                        )
                    )
                else:
                    new_past.append((layer[0], layer[1]))
            past_key_values = tuple(new_past)
        return past_key_values
