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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_ARRAY_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_ARRAY_OPS_DECLARE_H_

#include <string>
#include "utils/hash_map.h"
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "ops/array_ops.h"

namespace mindspore::transform {
DECLARE_OP_ADAPTER(Shape)//将shape收录后与标准进行比较，进行空间调整，用input_map_为key储存
DECLARE_OP_USE_OUTPUT(Shape)//将shape收录后与标准进行比较，进行空间调整，用output_map_为key储存
//下同，根据引入的不同变量进行对不同变量的操作

DECLARE_OP_ADAPTER(Reshape)
DECLARE_OP_USE_OUTPUT(Reshape)

DECLARE_OP_ADAPTER(TransShape)
DECLARE_OP_USE_INPUT_ATTR(TransShape)//将TransShape收录后与标准进行比较，进行空间调整，用input_map_为key储存
DECLARE_OP_USE_OUTPUT(TransShape)

DECLARE_OP_ADAPTER(MirrorPad)
DECLARE_OP_USE_OUTPUT(MirrorPad)

DECLARE_OP_ADAPTER(MirrorPadGrad)
DECLARE_OP_USE_OUTPUT(MirrorPadGrad)

DECLARE_OP_ADAPTER(ExpandDims)
DECLARE_OP_USE_OUTPUT(ExpandDims)

DECLARE_OP_ADAPTER(Squeeze)
DECLARE_OP_USE_OUTPUT(Squeeze)

DECLARE_OP_ADAPTER(Constant)
DECLARE_OP_USE_OUTPUT(Constant)

DECLARE_OP_ADAPTER(Summary)

DECLARE_OP_ADAPTER(Const)
DECLARE_OP_USE_OUTPUT(Const)

DECLARE_OP_ADAPTER(Data)

DECLARE_OP_ADAPTER(ReverseSequence)
DECLARE_OP_USE_OUTPUT(ReverseSequence)

DECLARE_OP_ADAPTER(EditDistance)
DECLARE_OP_USE_OUTPUT(EditDistance)

DECLARE_OP_ADAPTER(NonZero)
DECLARE_OP_USE_OUTPUT(NonZero)

DECLARE_OP_ADAPTER(Unsqueeze)
DECLARE_OP_USE_OUTPUT(Unsqueeze)
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_ARRAY_OPS_DECLARE_H_
