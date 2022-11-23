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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NPU_LOSS_SCALE_OPS_DECLARE_H_//判断宏是否被定义，如果宏没有定义，则编译下面代码
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NPU_LOSS_SCALE_OPS_DECLARE_H_
//定义预处理宏_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NPU_LOSS_SCALE_OPS_DECLARE_H_

#include <string>//导入标准库中的字符串类及相关操作
#include "utils/hash_map.h"//按照路径寻找以下文件，导入到本文件,以下同理
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "ops/npu_loss_scale_ops.h"

namespace mindspore::transform {//创建名为transform的空间，其空间处于空间mindspore下
//对实现NPUGetFloatStatus的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(NPUGetFloatStatus)
DECLARE_OP_USE_OUTPUT(NPUGetFloatStatus)
//对实现NPUAllocFloatStatus的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(NPUAllocFloatStatus)
DECLARE_OP_USE_OUTPUT(NPUAllocFloatStatus)
//对实现NPUClearFloatStatus的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(NPUClearFloatStatus)
DECLARE_OP_USE_OUTPUT(NPUClearFloatStatus)
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_NPU_LOSS_SCALE_OPS_DECLARE_H_
