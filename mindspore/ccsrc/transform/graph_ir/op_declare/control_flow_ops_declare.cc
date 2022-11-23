/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/control_flow_ops_declare.h"

namespace mindspore::transform {
// Merge
INPUT_MAP(Merge) = EMPTY_INPUT_MAP;//将Merge与标准进行比较，使原以input_map_为的key变为空
DYN_INPUT_MAP(Merge) = {{1, DYN_INPUT_DESC(x)}};//将Merge与标准进行比较，以dyn_input_map_为key并构造储存了name变量x的DYN_INPUT_DESC结构体并与标准比较其容量
ATTR_MAP(Merge) = EMPTY_ATTR_MAP;//将Merge与标准进行比较，使原以input_map_为的key变为空
OUTPUT_MAP(Merge) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(value_index)}};//将Merge与标准进行比较，并分别将两组变量与标准比较其容量
REG_ADPT_DESC(Merge, kNameMerge, ADPT_DESC(Merge))//将Merge处理并存入对应OutputDesc结构体的相应变量中
  //将Merge内容转为字符串变量并存储至结构体的name变量中
  //引用Operator空间并将指针所指的类转为OutputDesc结构体

// Switch
INPUT_MAP(Switch) = {{1, INPUT_DESC(data)}, {2, INPUT_DESC(pred)}};
OUTPUT_MAP(Switch) = {{0, OUTPUT_DESC(output_false)}, {1, OUTPUT_DESC(output_true)}};
ATTR_MAP(Switch) = EMPTY_ATTR_MAP;
REG_ADPT_DESC(Switch, kNameGeSwitch, ADPT_DESC(Switch))
}  // namespace mindspore::transform
