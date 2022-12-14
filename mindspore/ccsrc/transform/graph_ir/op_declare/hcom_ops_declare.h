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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_HCOM_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_HCOM_OPS_DECLARE_H_

#include <string>
#include "utils/hash_map.h"
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "ops/hcom_ops.h"

namespace mindspore::transform {
DECLARE_OP_ADAPTER(HcomReduceScatter)//将HcomReduceScatter的类型与标准进行对比，后进行空间调整，并用指针input_map_为key存储相应内容
DECLARE_OP_USE_OUTPUT(HcomReduceScatter)//将HcomReduceScatter的类型与标准进行对比，后进行空间调整，并用指针output_map_为key存储相应内容

DECLARE_OP_ADAPTER(HcomBroadcast)
DECLARE_OP_USE_DYN_INPUT(HcomBroadcast)//将HcomReduceScatter的类型与标准进行对比，后进行空间调整，并用指针dyn_input_map_为key存储相应内容
DECLARE_OP_USE_DYN_OUTPUT(HcomBroadcast)//将HcomReduceScatter的类型与标准进行对比，后进行空间调整，并用指针dyn_output_map_为key存储相应内容

DECLARE_OP_ADAPTER(HcomAllReduce)
DECLARE_OP_USE_OUTPUT(HcomAllReduce)

DECLARE_OP_ADAPTER(HcomAllGather)
DECLARE_OP_USE_OUTPUT(HcomAllGather)
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_HCOM_OPS_DECLARE_H_
