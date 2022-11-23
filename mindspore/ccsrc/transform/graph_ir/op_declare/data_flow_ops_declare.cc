/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/data_flow_ops_declare.h"
#include <vector>

namespace mindspore::transform {
INPUT_MAP(TensorArray) = {{1, INPUT_DESC(size)}};/*
    将value处理并存入对应ATTR_DESC结构体的相应变量中
  将value内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为ATTR_DESC结构体
将TensorArray的类型与标准进行对比，后进行空间调整，并用指针input_map_为key存储相应内容
*/
ATTR_MAP(TensorArray) = {{"dtype", ATTR_DESC(dtype, AnyTraits<GEType>())},/*
  将dtype处理并存入对应ATTR_DESC结构体的相应变量中
  将dtype内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为ATTR_DESC结构体
//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//存入MaxPool对应空间并用attr_map_指针保存
//其中AnyTraits<>的作用为将<>内类型进行构建*/
                         {"element_shape", ATTR_DESC(element_shape, AnyTraits<std::vector<int64_t>>())},
                         {"dynamic_size", ATTR_DESC(dynamic_size, AnyTraits<bool>())},
                         {"clear_after_read", ATTR_DESC(clear_after_read, AnyTraits<bool>())},
                         {"identical_element_shapes", ATTR_DESC(identical_element_shapes, AnyTraits<bool>())},
                         {"tensor_array_name", ATTR_DESC(tensor_array_name, AnyTraits<std::string>())}};//将TensorArray的类型与标准进行对比，后进行空间调整，并用指针attr_map_为key存储相应内容
OUTPUT_MAP(TensorArray) = {{0, OUTPUT_DESC(handle)}, {1, OUTPUT_DESC(flow)}};/*
  将handle和flow处理并存入对应ATTR_DESC结构体的相应变量中
  将变量内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为OUTPUT_DESC结构体*/
//将TensorArray的类型与标准进行对比，后进行空间调整，并用指针output_map_为key存储相应内容
REG_ADPT_DESC(TensorArray, kNameTensorArray, ADPT_DESC(TensorArray))/*
  将TensorArray处理并存入对应ADPT_DESC结构体的相应变量中
  将TensorArray内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为ADPT_DESC结构体
  
  再将TensorArray处理并存入对应REG_ADPT_DESC结构体的相应变量中
  将TensorArray内容转为字符串变量并存储至REG_ADPT_DESC的结构体的name变量中
  引用Operator空间并将指针所指的类转为REG_ADPT_DESC结构体
*/

INPUT_MAP(TensorArrayWrite) = {
  {1, INPUT_DESC(handle)}, {2, INPUT_DESC(index)}, {3, INPUT_DESC(value)}, {4, INPUT_DESC(flow_in)}};
ATTR_MAP(TensorArrayWrite) = EMPTY_ATTR_MAP;
OUTPUT_MAP(TensorArrayWrite) = {{0, OUTPUT_DESC(flow_out)}};
REG_ADPT_DESC(TensorArrayWrite, kNameTensorArrayWrite, ADPT_DESC(TensorArrayWrite))

INPUT_MAP(TensorArrayGather) = {{1, INPUT_DESC(handle)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(flow_in)}};
ATTR_MAP(TensorArrayGather) = {{"dtype", ATTR_DESC(dtype, AnyTraits<GEType>())},
                               {"element_shape", ATTR_DESC(element_shape, AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(TensorArrayGather) = {{0, OUTPUT_DESC(value)}};
REG_ADPT_DESC(TensorArrayGather, kNameTensorArrayGather, ADPT_DESC(TensorArrayGather))
}  // namespace mindspore::transform
