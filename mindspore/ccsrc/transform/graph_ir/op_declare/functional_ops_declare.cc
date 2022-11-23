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

#include "transform/graph_ir/op_declare/functional_ops_declare.h"

namespace mindspore::transform {
// Case
INPUT_MAP(Case) = {{1, INPUT_DESC(branch_index)}};/*
  将branch_index处理并存入对应INPUT_DESC结构体的相应变量中
  将branch_index内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为INPUT_DESC结构体
将Case的类型与标准进行对比，后进行空间调整，并用指针input_map_为key存储相应内容
*/
DYN_INPUT_MAP(Case) = {{2, DYN_INPUT_DESC(input)}};/*
  将branch_index处理并存入对应DYN_INPUT_DESC结构体的相应变量中
  将branch_index内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为DYN_INPUT_DESC结构体
将Case的类型与标准进行对比，后进行空间调整，并用指针dyn_input_map_为key存储相应内容
*/
ATTR_MAP(Case) = EMPTY_ATTR_MAP;//将Case与标准进行比较，使原以attr_map_为储存信息的key变为空
DYN_OUTPUT_MAP(Case) = {{0, DYN_OUTPUT_DESC(output)}};/*
  将(output处理并存入对应DYN_OUTPUT_DESC结构体的相应变量中
  将(output内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为DYN_OUTPUT_DESC结构体
将Case的类型与标准进行对比，后进行空间调整，并用指针dyn_output_map_为key存储相应内容
*/
DYN_SUBGRAPH_MAP(Case) = {{0, DYN_SUBGRAPH_DESC(branches)}};/*
  将branches处理并存入对应DYN_SUBGRAPH_DESC结构体的相应变量中
  将branches内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为DYN_SUBGRAPH_DESC结构体
将Case的类型与标准进行对比，后进行空间调整，并用指针dyn_subgrraph_map_为key存储相应内容
*/
REG_ADPT_DESC(Case, kNameCase, ADPT_DESC(Case))/*
  将Case处理并存入对应ADPT_DESC结构体的相应变量中
  将Case内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为ADPT_DESC结构体
  
  再将Case处理并存入对应REG_ADPT_DESC结构体的相应变量中
  将Case内容转为字符串变量并存储至REG_ADPT_DESC的结构体的name变量中
  引用Operator空间并将指针所指的类转为REG_ADPT_DESC结构体
*/
}  // namespace mindspore::transform
