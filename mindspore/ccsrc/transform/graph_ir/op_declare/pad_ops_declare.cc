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

#include "transform/graph_ir/op_declare/pad_ops_declare.h"//按照路径寻找以下文件，导入到本文件
#include <vector>//提供vector数组构建函数模版等

namespace mindspore::transform {//创建名为transform的空间，其空间处于空间mindspore下
// PadD
INPUT_MAP(PadD) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入NPUGetFloatStatus对应空间内并用input_map指针保存
ATTR_MAP(PadD) = {{"paddings", ATTR_DESC(paddings, AnyTraits<std::vector<std::vector<int64_t>>>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                                     存入PadD对应空间并用attr_map_指针保存
//                                                                                                     其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                                                     std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(PadD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入PadD对应空间内并用output_map_指针保存
REG_ADPT_DESC(PadD, kNamePadD, ADPT_DESC(PadD))//构造指向PadD的指针并储存，创建结构体RegAdptDescPadD

// Pad
INPUT_MAP(Pad) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(paddings)}};
//将变量x、paddings处理并存入对应InputDesc结构体的相应变量中，存入Pad对应空间内并用input_map指针保存
ATTR_MAP(Pad) = EMPTY_ATTR_MAP;//将空变量存入Pad对应空间并用attr_map_指针保存
OUTPUT_MAP(Pad) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入Pad对应空间内并用output_map_指针保存
REG_ADPT_DESC(Pad, kNamePadV1, ADPT_DESC(Pad))//构造指向Pad的指针并储存，创建结构体RegAdptDescPad

// BroadcastToD
INPUT_MAP(BroadcastToD) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入BroadcastToD对应空间内并用input_map指针保存
ATTR_MAP(BroadcastToD) = {{"shape", ATTR_DESC(shape, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())}};
//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//存入BroadcastToD对应空间并用attr_map_指针保存
//其中AnyTraits<>的作用为将<>内类型进行构建
//std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(BroadcastToD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入BroadcastToD对应空间内并用output_map_指针保存
REG_ADPT_DESC(BroadcastToD, kNameBroadcastTo, ADPT_DESC(BroadcastToD))//构造指向BroadcastToD的指针并储存，创建结构体RegAdptDescBroadcastToD

// Diag
INPUT_MAP(Diag) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入Diag对应空间内并用input_map指针保存
ATTR_MAP(Diag) = EMPTY_ATTR_MAP;//将空变量存入Diag对应空间并用attr_map_指针保存
OUTPUT_MAP(Diag) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入Diag对应空间内并用output_map_指针保存
REG_ADPT_DESC(Diag, kNameDiag, ADPT_DESC(Diag))//构造指向Diag的指针并储存，创建结构体RegAdptDescDiag

// FillD
INPUT_MAP(FillD) = {{1, INPUT_DESC(value)}};//将变量value处理并存入对应InputDesc结构体的相应变量中，存入FillD对应空间内并用input_map指针保存
ATTR_MAP(FillD) = {{"dims", ATTR_DESC(dims, AnyTraits<std::vector<int64_t>>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                 存入FillD对应空间并用attr_map_指针保存
//                                                                                 其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                                 std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(FillD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入FillD对应空间内并用output_map_指针保存
REG_ADPT_DESC(FillD, kNameFillD, ADPT_DESC(FillD))//构造指向FillD的指针并储存，创建结构体RegAdptDescFillD

// Fill
INPUT_MAP(Fill) = {{1, INPUT_DESC(dims)}, {2, INPUT_DESC(value)}};
//将变量dims、value处理并存入对应InputDesc结构体的相应变量中，存入Fill对应空间内并用input_map指针保存
ATTR_MAP(Fill) = EMPTY_ATTR_MAP;//将空变量存入Fill对应空间并用attr_map_指针保存
OUTPUT_MAP(Fill) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入Fill对应空间内并用output_map_指针保存
REG_ADPT_DESC(Fill, kNameFillV1, ADPT_DESC(Fill))//构造指向Fill的指针并储存，创建结构体RegAdptDescFill

// PadV3
INPUT_MAP(PadV3) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(paddings)}, {3, INPUT_DESC(constant_values)}};
//将变量x、paddings、constant_values处理并存入对应InputDesc结构体的相应变量中，存入PadV3对应空间内并用input_map指针保存
ATTR_MAP(PadV3) = {{"mode", ATTR_DESC(mode, AnyTraits<std::string>())},
                   {"pad_contiguous", ATTR_DESC(paddings_contiguous, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                          存入PadV3对应空间并用attr_map_指针保存
//                                                                                          其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                                          std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(PadV3) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入PadV3对应空间内并用output_map_指针保存
REG_ADPT_DESC(PadV3, kNamePadV3, ADPT_DESC(PadV3))//构造指向PadV3的指针并储存，创建结构体RegAdptDescPadV3

// PadV2
INPUT_MAP(PadV2) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(paddings)}, {3, INPUT_DESC(constant_values)}};
//将变量x、paddings、constant_values处理并存入对应InputDesc结构体的相应变量中，存入PadV2对应空间内并用input_map指针保存
ATTR_MAP(PadV2) = EMPTY_ATTR_MAP;//将空变量存入PadV2对应空间并用attr_map_指针保存
OUTPUT_MAP(PadV2) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入PadV2对应空间内并用output_map_指针保存
REG_ADPT_DESC(PadV2, kNamePadV2, ADPT_DESC(PadV2))//构造指向PadV2的指针并储存，创建结构体RegAdptDescPadV2
}  // namespace mindspore::transform
