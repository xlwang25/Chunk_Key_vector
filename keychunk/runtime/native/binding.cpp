#include <torch/extension.h>
#include "position.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {m.def("rotary_embedding_k_fused", &rotary_embedding_k_fused);}
