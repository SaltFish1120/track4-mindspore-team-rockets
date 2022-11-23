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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_POOLING_OPS_DECLARE_H_//判断宏是否被定义，如果宏没有定义，则编译下面代码
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_POOLING_OPS_DECLARE_H_//定义预处理宏_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_POOLING_OPS_DECLARE_H_

#include <string>//导入标准库中的字符串类及相关操作
#include "utils/hash_map.h"//按照路径寻找以下文件，导入到本文件,以下同理
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "ops/nn_ops.h"
#include "ops/nn_pooling_ops.h"

namespace mindspore::transform {//创建名为transform的空间，其空间处于空间mindspore下
//对实现MaxPoolWithArgmax的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(MaxPoolWithArgmax)
DECLARE_OP_USE_OUTPUT(MaxPoolWithArgmax)
//对实现MaxPoolGradWithArgmax的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(MaxPoolGradWithArgmax)
DECLARE_OP_USE_OUTPUT(MaxPoolGradWithArgmax)
//对实现MaxPoolGradGradWithArgmax的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(MaxPoolGradGradWithArgmax)
DECLARE_OP_USE_OUTPUT(MaxPoolGradGradWithArgmax)
//对实现MaxPool的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(MaxPool)
DECLARE_OP_USE_OUTPUT(MaxPool)
//对实现MaxPoolGrad的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(MaxPoolGrad)
DECLARE_OP_USE_OUTPUT(MaxPoolGrad)
//对实现MaxPoolGradGrad的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(MaxPoolGradGrad)
DECLARE_OP_USE_OUTPUT(MaxPoolGradGrad)
//对实现MaxPool3D的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(MaxPool3D)
DECLARE_OP_USE_OUTPUT(MaxPool3D)
//对实现MaxPool3DGrad的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(MaxPool3DGrad)
DECLARE_OP_USE_OUTPUT(MaxPool3DGrad)
//对实现MaxPool3DGradGrad的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(MaxPool3DGradGrad)
DECLARE_OP_USE_OUTPUT(MaxPool3DGradGrad)
//对实现AvgPool的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(AvgPool)
DECLARE_OP_USE_OUTPUT(AvgPool)
//对实现AvgPoolGrad的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(AvgPoolGrad)
DECLARE_OP_USE_OUTPUT(AvgPoolGrad)
//对实现Pooling的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(Pooling)
DECLARE_OP_USE_OUTPUT(Pooling)
//对实现MaxPoolV3的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(MaxPoolV3)
DECLARE_OP_USE_OUTPUT(MaxPoolV3)
//对实现AvgPoolV2的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(AvgPoolV2)
DECLARE_OP_USE_OUTPUT(AvgPoolV2)
//对实现GlobalAveragePool的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(GlobalAveragePool)
DECLARE_OP_USE_OUTPUT(GlobalAveragePool)
//对实现Upsample的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(Upsample)
DECLARE_OP_USE_OUTPUT(Upsample)
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NN_POOLING_OPS_DECLARE_H_
