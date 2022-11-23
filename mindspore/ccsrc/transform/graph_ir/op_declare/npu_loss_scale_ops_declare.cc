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

#include "transform/graph_ir/op_declare/npu_loss_scale_ops_declare.h"//按照路径寻找以下文件，导入到本文件

namespace mindspore::transform {//创建名为transform的空间，其空间处于空间mindspore下
// NPUGetFloatStatus
INPUT_MAP(NPUGetFloatStatus) = {{1, INPUT_DESC(addr)}};//将变量addr处理并存入对应InputDesc结构体的相应变量中，存入NPUGetFloatStatus对应空间内并用input_map指针保存
OUTPUT_MAP(NPUGetFloatStatus) = {{0, OUTPUT_DESC(data)}};
//将变量data处理并存入对应OutputDesc结构体的相应变量中，存入NPUGetFloatStatus对应空间内并用output_map_指针保存
ATTR_MAP(NPUGetFloatStatus) = EMPTY_ATTR_MAP;//将空变量存入NPUGetFloatStatus对应空间并用attr_map_指针保存
REG_ADPT_DESC(NPUGetFloatStatus, kNameNPUGetFloatStatus, ADPT_DESC(NPUGetFloatStatus))
//构造指向NPUGetFloatStatus的指针并储存，创建结构体RegAdptDescNPUGetFloatStatus

// NPUAllocFloatStatus
INPUT_MAP(NPUAllocFloatStatus) = EMPTY_INPUT_MAP;//将空变量存入NPUAllocFloatStatus对应空间内并用input_map指针保存
ATTR_MAP(NPUAllocFloatStatus) = EMPTY_ATTR_MAP;//将空变量存入NPUAllocFloatStatus对应空间并用attr_map_指针保存
OUTPUT_MAP(NPUAllocFloatStatus) = {{0, OUTPUT_DESC(data)}};
//将变量data处理并存入对应OutputDesc结构体的相应变量中，存入NPUAllocFloatStatus对应空间内并用output_map_指针保存
REG_ADPT_DESC(NPUAllocFloatStatus, kNameNPUAllocFloatStatus, ADPT_DESC(NPUAllocFloatStatus))
//构造指向NPUAllocFloatStatus的指针并储存，创建结构体RegAdptDescNPUAllocFloatStatus

// NPUClearFloatStatus
INPUT_MAP(NPUClearFloatStatus) = {{1, INPUT_DESC(addr)}};//将变量addr处理并存入对应InputDesc结构体的相应变量中，存入NPUClearFloatStatus对应空间内并用input_map指针保存
OUTPUT_MAP(NPUClearFloatStatus) = {{0, OUTPUT_DESC(data)}};
//将变量data处理并存入对应OutputDesc结构体的相应变量中，存入NPUClearFloatStatus对应空间内并用output_map_指针保存
ATTR_MAP(NPUClearFloatStatus) = EMPTY_ATTR_MAP;//将空变量存入NPUClearFloatStatus对应空间并用attr_map_指针保存
REG_ADPT_DESC(NPUClearFloatStatus, kNameNPUClearFloatStatus, ADPT_DESC(NPUClearFloatStatus))
//构造指向NPUClearFloatStatus的指针并储存，创建结构体RegAdptDescNPUClearFloatStatus
}  // namespace mindspore::transform
