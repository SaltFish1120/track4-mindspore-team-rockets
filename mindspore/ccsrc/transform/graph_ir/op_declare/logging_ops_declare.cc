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

#include "transform/graph_ir/op_declare/logging_ops_declare.h"

namespace mindspore::transform {
// Print
INPUT_MAP(Print) = EMPTY_INPUT_MAP;//将Print的类型与标准进行对比，后进行空间调整，使原input_map_为储存相应内容的key为空
DYN_INPUT_MAP(Print) = {{1, DYN_INPUT_DESC(x)}};/*
    将x处理并存入对应DYN_INPUT_DESC结构体的相应变量中
  将x内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为DYN_INPUT_DESC结构体
将Print的类型与标准进行对比，后进行空间调整，并用指针dyn_input_map_为key存储相应内容
*/
ATTR_MAP(Print) = EMPTY_ATTR_MAP;//将TensorScatterUpdate的类型与标准进行对比，后进行空间调整，使原attr_map_为储存相应内容的key为空
REG_ADPT_DESC(Print, kNamePrint, ADPT_DESC(Print))/*
  将Print处理并存入对应ADPT_DESC结构体的相应变量中
  将Print内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为ADPT_DESC结构体
  
  再将Print处理并存入对应REG_ADPT_DESC结构体的相应变量中
  将Print内容转为字符串变量并存储至REG_ADPT_DESC的结构体的name变量中
  引用Operator空间并将指针所指的类转为REG_ADPT_DESC结构体
*/          


#ifdef ENABLE_D
INPUT_MAP(Assert) = {{1, INPUT_DESC(input_condition)}};
DYN_INPUT_MAP(Assert) = {{2, DYN_INPUT_DESC(input_data)}};
ATTR_MAP(Assert) = {{"summarize", ATTR_DESC(summarize, AnyTraits<int64_t>())}};
REG_ADPT_DESC(Assert, kNameAssert, ADPT_DESC(Assert))
#endif
}  // namespace mindspore::transform
