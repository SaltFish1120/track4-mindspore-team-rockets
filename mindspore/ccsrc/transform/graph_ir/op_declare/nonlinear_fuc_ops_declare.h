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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NONLINEAR_FUC_OPS_DECLARE_H_//判断宏是否被定义，如果宏没有定义，则编译下面代码
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NONLINEAR_FUC_OPS_DECLARE_H_//定义预处理宏_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NONLINEAR_FUC_OPS_DECLARE_H_

#include <string>//导入标准库中的字符串类及相关操作
#include "utils/hash_map.h"//按照路径寻找以下文件，导入到本文件,以下同理
#include "ops/nonlinear_fuc_ops.h"
#include "transform/graph_ir/op_declare/op_declare_macro.h"

namespace mindspore::transform {//创建名为transform的空间，其空间处于空间mindspore下
//对实现ReluGrad的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(ReluGrad)
DECLARE_OP_USE_OUTPUT(ReluGrad)
//对实现ReluGradV2的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(ReluGradV2)
DECLARE_OP_USE_OUTPUT(ReluGradV2)
//对实现Relu6的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(Relu6)
DECLARE_OP_USE_OUTPUT(Relu6)
//对实现Relu6Grad的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(Relu6Grad)
DECLARE_OP_USE_OUTPUT(Relu6Grad)
//对实现Softsign的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(Softsign)
DECLARE_OP_USE_OUTPUT(Softsign)
//对实现Softplus的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(Softplus)
DECLARE_OP_USE_OUTPUT(Softplus)
//对实现SoftplusGrad的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(SoftplusGrad)
DECLARE_OP_USE_OUTPUT(SoftplusGrad)
//对实现Tanh的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(Tanh)
DECLARE_OP_USE_OUTPUT(Tanh)
//对实现TanhGrad的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(TanhGrad)
DECLARE_OP_USE_OUTPUT(TanhGrad)
//对实现Mish的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(Mish)
DECLARE_OP_USE_OUTPUT(Mish)
//对实现Gelu的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(Gelu)
DECLARE_OP_USE_OUTPUT(Gelu)
//对实现GeluGrad的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(GeluGrad)
DECLARE_OP_USE_OUTPUT(GeluGrad)
//对实现FastGelu的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(FastGelu)
DECLARE_OP_USE_OUTPUT(FastGelu)
//对实现FastGeluGrad的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(FastGeluGrad)
DECLARE_OP_USE_OUTPUT(FastGeluGrad)
//对实现Relu的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(Relu)
DECLARE_OP_USE_OUTPUT(Relu)
//对实现ReluV2的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(ReluV2)
DECLARE_OP_USE_OUTPUT(ReluV2)
//对实现PRelu的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(PRelu)
DECLARE_OP_USE_OUTPUT(PRelu)
//对实现Elu的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(Elu)
DECLARE_OP_USE_OUTPUT(Elu)
//对实现EluGrad的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(EluGrad)
DECLARE_OP_USE_OUTPUT(EluGrad)
//对实现PReluGrad的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(PReluGrad)
DECLARE_OP_USE_OUTPUT(PReluGrad)
//对实现Selu的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(Selu)
DECLARE_OP_USE_OUTPUT(Selu)
//对实现Sigmoid的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(Sigmoid)
DECLARE_OP_USE_OUTPUT(Sigmoid)
//对实现HardSwish的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(HardSwish)
DECLARE_OP_USE_OUTPUT(HardSwish)
//对实现HardSwishGrad的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(HardSwishGrad)
DECLARE_OP_USE_OUTPUT(HardSwishGrad)
//对实现HardSigmoid的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(HardSigmoid)
DECLARE_OP_USE_OUTPUT(HardSigmoid)
//对实现SigmoidGrad的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(SigmoidGrad)
DECLARE_OP_USE_OUTPUT(SigmoidGrad)
//对实现LeakyRelu的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(LeakyRelu)
DECLARE_OP_USE_OUTPUT(LeakyRelu)
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NONLINEAR_FUC_OPS_DECLARE_H_
