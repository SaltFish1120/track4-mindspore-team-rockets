# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""LayerNormBetaGammaBackprop op"""
from mindspore.ops.op_info_register import op_info_register, TBERegOp, DataType

layer_norm_beta_gamma_backprop_v2_op_info = TBERegOp("LayerNormBetaGammaBackpropV2") \
    .fusion_type("OPAQUE") \
    .async_flag(False) \
    .binfile_name("layer_norm_beta_gamma_backprop_v2.so") \
    .compute_cost(10) \
    .kernel_name("layer_norm_beta_gamma_backprop_v2") \
    .partial_flag(True) \
    .attr("shape_gamma", "required", "listInt", "all") \
    .input(0, "dy", False, "required", "all") \
    .input(1, "res_for_gamma", False, "required", "all") \
    .output(0, "pd_gamma", False, "required", "all") \
    .output(1, "pd_beta", False, "required", "all") \
    .is_dynamic_format(True) \
    .dtype_format(DataType.F16_None, DataType.F32_None, DataType.F32_None, DataType.F32_None) \
    .dtype_format(DataType.F32_None, DataType.F32_None, DataType.F32_None, DataType.F32_None) \
    .get_op_info()


@op_info_register(layer_norm_beta_gamma_backprop_v2_op_info)
def _layer_norm_beta_gamma_backprop_v2_tbe():
    """LayerNormBetaGammaBackpropV2 TBE register"""
    return
