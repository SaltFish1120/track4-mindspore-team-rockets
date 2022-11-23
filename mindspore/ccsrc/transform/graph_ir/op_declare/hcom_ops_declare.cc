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

#include "transform/graph_ir/op_declare/hcom_ops_declare.h"

namespace mindspore::transform {
// HCOMAllreduce
INPUT_MAP(HcomAllReduce) = {{1, INPUT_DESC(x)}};/*
    将x处理并存入对应INPUT_DESC结构体的相应变量中
  将x内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为INPUT_DESC结构体
将HcomAllReduce的类型与标准进行对比，后进行空间调整，并用指针input_map_为key存储相应内容
*/
OUTPUT_MAP(HcomAllReduce) = {{0, OUTPUT_DESC(y)}};/*
    将y处理并存入对应OUTPUT_DESC结构体的相应变量中
  将x内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为OUTPUT_DESC结构体
将(HcomAllReduce的类型与标准进行对比，后进行空间调整，并用指针output_map_为key存储相应内容
*/
ATTR_MAP(HcomAllReduce) = {{"op", ATTR_DESC(reduction, AnyTraits<std::string>())},
                           {"group", ATTR_DESC(group, AnyTraits<std::string>())},
                           {"fusion", ATTR_DESC(fusion, AnyTraits<int64_t>())}};/*
  将reduction处理并存入对应ATTR_DESC结构体的相应变量中
  将reduction内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为ATTR_DESC结构体
//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//存入MaxPool对应空间并用attr_map_指针保存
//其中AnyTraits<>的作用为将<>内类型进行构建
将HcomAllReduce的类型与标准进行对比，后进行空间调整，并用指针attr_map_为key存储相应内容
*/
REG_ADPT_DESC(HcomAllReduce, kNameAllReduce, ADPT_DESC(HcomAllReduce))/*
  将HcomAllReduce处理并存入对应ADPT_DESC结构体的相应变量中
  将HcomAllReduce内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为ADPT_DESC结构体
  
  再将HcomAllReduce处理并存入对应REG_ADPT_DESC结构体的相应变量中
  将HcomAllReduce内容转为字符串变量并存储至REG_ADPT_DESC的结构体的name变量中
  引用Operator空间并将指针所指的类转为REG_ADPT_DESC结构体
*/          

// HCOMBraodcast
INPUT_MAP(HcomBroadcast) = EMPTY_INPUT_MAP;
DYN_INPUT_MAP(HcomBroadcast) = {{1, DYN_INPUT_DESC(x)}};
DYN_OUTPUT_MAP(HcomBroadcast) = {{0, DYN_OUTPUT_DESC(y)}};
ATTR_MAP(HcomBroadcast) = {{"root_rank", ATTR_DESC(root_rank, AnyTraits<int64_t>())},
                           {"group", ATTR_DESC(group, AnyTraits<std::string>())}};
REG_ADPT_DESC(HcomBroadcast, kNameBroadcast, ADPT_DESC(HcomBroadcast))

// HcomAllGather
INPUT_MAP(HcomAllGather) = {{1, INPUT_DESC(x)}};
OUTPUT_MAP(HcomAllGather) = {{0, OUTPUT_DESC(y)}};
ATTR_MAP(HcomAllGather) = {{"group", ATTR_DESC(group, AnyTraits<std::string>())},
                           {"rank_size", ATTR_DESC(rank_size, AnyTraits<int64_t>())}};
REG_ADPT_DESC(HcomAllGather, kNameAllgather, ADPT_DESC(HcomAllGather))

// HCOMReduceScatter
INPUT_MAP(HcomReduceScatter) = {{1, INPUT_DESC(x)}};
OUTPUT_MAP(HcomReduceScatter) = {{0, OUTPUT_DESC(y)}};
ATTR_MAP(HcomReduceScatter) = {{"group", ATTR_DESC(group, AnyTraits<std::string>())},
                               {"op", ATTR_DESC(reduction, AnyTraits<std::string>())},
                               {"rank_size", ATTR_DESC(rank_size, AnyTraits<int64_t>())}};
REG_ADPT_DESC(HcomReduceScatter, kNameReduceScatter, ADPT_DESC(HcomReduceScatter))
}  // namespace mindspore::transform
