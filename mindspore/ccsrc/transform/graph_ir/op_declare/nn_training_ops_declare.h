/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_TRAINING_OPS_DECLARE_H_//判断宏是否被定义，如果宏没有定义，则编译下面代码
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_TRAINING_OPS_DECLARE_H_//定义预处理宏_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_TRAINING_OPS_DECLARE_H_

#include <string>//导入标准库中的字符串类及相关操作
#include "utils/hash_map.h"//按照路径寻找以下文件，导入到本文件,以下同理
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "ops/nn_training_ops.h"

namespace mindspore::transform {//创建名为transform的空间，其空间处于空间mindspore下
//对实现ApplyAdam的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(ApplyAdam)
DECLARE_OP_USE_OUTPUT(ApplyAdam)
//对实现ApplyAdamD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(ApplyAdamD)
DECLARE_OP_USE_OUTPUT(ApplyAdamD)
//对实现ApplyAdagradD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(ApplyAdagradD)
DECLARE_OP_USE_OUTPUT(ApplyAdagradD)
//对实现ApplyAdagradV2D的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(ApplyAdagradV2D)
DECLARE_OP_USE_OUTPUT(ApplyAdagradV2D)
//对实现ApplyAddSignD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(ApplyAddSignD)
DECLARE_OP_USE_OUTPUT(ApplyAddSignD)
//对实现SparseApplyAdagradV2D的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(SparseApplyAdagradV2D)
DECLARE_OP_USE_OUTPUT(SparseApplyAdagradV2D)
//对实现DataFormatDimMap的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(DataFormatDimMap)
DECLARE_OP_USE_OUTPUT(DataFormatDimMap)
//对实现ApplyAdadeltaD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(ApplyAdadeltaD)
DECLARE_OP_USE_OUTPUT(ApplyAdadeltaD)
//对实现ApplyAdaMaxD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(ApplyAdaMaxD)
DECLARE_OP_USE_OUTPUT(ApplyAdaMaxD)
//对实现ApplyGradientDescent的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(ApplyGradientDescent)
DECLARE_OP_USE_OUTPUT(ApplyGradientDescent)
//对实现ApplyPowerSignD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(ApplyPowerSignD)
DECLARE_OP_USE_OUTPUT(ApplyPowerSignD)
//对实现ApplyProximalGradientDescent的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(ApplyProximalGradientDescent)
DECLARE_OP_USE_OUTPUT(ApplyProximalGradientDescent)
//对实现SGD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(SGD)
DECLARE_OP_USE_OUTPUT(SGD)
//对实现ApplyMomentum的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(ApplyMomentum)
DECLARE_OP_USE_OUTPUT(ApplyMomentum)
//对实现SparseApplyAdagradD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(SparseApplyAdagradD)
DECLARE_OP_USE_OUTPUT(SparseApplyAdagradD)
//对实现ApplyProximalAdagradD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(ApplyProximalAdagradD)
DECLARE_OP_USE_OUTPUT(ApplyProximalAdagradD)
//对实现SparseApplyProximalAdagradD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(SparseApplyProximalAdagradD)
DECLARE_OP_USE_OUTPUT(SparseApplyProximalAdagradD)
//对实现LarsV2Update的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(LarsV2Update)
DECLARE_OP_USE_OUTPUT(LarsV2Update)
//对实现ApplyFtrl的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(ApplyFtrl)
DECLARE_OP_USE_OUTPUT(ApplyFtrl)
//对实现SparseApplyFtrlD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(SparseApplyFtrlD)
DECLARE_OP_USE_OUTPUT(SparseApplyFtrlD)
//对实现SparseApplyFtrlV2D的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(SparseApplyFtrlV2D)
DECLARE_OP_USE_OUTPUT(SparseApplyFtrlV2D)
//对实现ApplyRMSPropD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(ApplyRMSPropD)
DECLARE_OP_USE_INPUT_ATTR(ApplyRMSPropD)
DECLARE_OP_USE_OUTPUT(ApplyRMSPropD)
//对实现ApplyCenteredRMSProp的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(ApplyCenteredRMSProp)
DECLARE_OP_USE_OUTPUT(ApplyCenteredRMSProp)
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_TRAINING_OPS_DECLARE_H_
