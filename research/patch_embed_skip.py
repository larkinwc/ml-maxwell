#!/usr/bin/env python3
F="/home/larkinwc/maxwell-stack/vllm-maxwell-core/vllm/model_executor/layers/quantization/gguf.py"
s=open(F).read()
if "Embeddings use ggml_dequantize" in s:
    print("already"); raise SystemExit
old='''class GGUFEmbeddingMethod(GGUFLinearMethod):
    """Embedding method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:'''
new='''class GGUFEmbeddingMethod(GGUFLinearMethod):
    """Embedding method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module):
        # Embeddings use ggml_dequantize (not MMVQ/SoA) and read qweight.shape
        # directly, so never build SoA buffers or free the AoS qweight here.
        qweight_type = layer.qweight_type.weight_type
        if not (qweight_type in UNQUANTIZED_TYPES or qweight_type in DEQUANT_TYPES):
            qweight_type = WeightType(qweight_type)
            raise ValueError(
                f"Unsupported GGUF quantization type {qweight_type} in layer {layer}."
            )
        self._create_padded_weight_param(layer)

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:'''
assert s.count(old)==1, s.count(old)
open(F,"w").write(s.replace(old,new,1))
print("embedding override added")
