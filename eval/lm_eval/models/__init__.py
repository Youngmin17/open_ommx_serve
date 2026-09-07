import importlib as _importlib
import logging as _logging

_log = _logging.getLogger(__name__)

# Import each model adapter INDEPENDENTLY so a missing optional backend (e.g. the
# kvquant adapter's `kvquant.simquant_module_quantizer`, or ommx's `quantizer_module`)
# doesn't take down the whole model registry / `python -m lm_eval` CLI. Adapters that
# import cleanly still register; broken ones are skipped with a warning.
for _m in (
    "anthropic_llms", "api_models", "dummy", "gguf", "hf_vlms", "huggingface",
    "ibm_watsonx_ai", "kivi_model", "kitty_model", "ommx_hf_model", "mamba_lm", "nemo_lm", "neuralmagic",
    "neuron_optimum", "openai_completions", "optimum_ipex", "optimum_lm",
    "sglang_causallms", "sglang_generate_API", "textsynth", "vllm_causallms",
    "vllm_vlms", "kvquant_adapter",
):
    try:
        _importlib.import_module(f".{_m}", __name__)
    except Exception as _e:  # noqa: BLE001
        _log.warning("lm_eval.models: skipped adapter %s (%s: %s)",
                     _m, type(_e).__name__, _e)


# TODO: implement __all__


try:
    # enable hf hub transfer if available
    import hf_transfer  # type: ignore # noqa
    import huggingface_hub.constants  # type: ignore

    huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
except ImportError:
    pass
