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

#include "transform/graph_ir/op_declare/random_ops_declare.h"//按照路径寻找以下文件，导入到本文件

namespace mindspore::transform {//创建名为transform的空间，其空间处于空间mindspore下
// DropOutGenMask
INPUT_MAP(DropOutGenMask) = {{1, INPUT_DESC(shape)}, {2, INPUT_DESC(prob)}};
//将变量shape、prob处理并存入对应InputDesc结构体的相应变量中，存入DropOutGenMask对应空间内并用input_map指针保存
ATTR_MAP(DropOutGenMask) = {{"Seed0", ATTR_DESC(seed, AnyTraits<int64_t>())},
                            {"Seed1", ATTR_DESC(seed2, AnyTraits<int64_t>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                               存入DropOutGenMask对应空间并用attr_map_指针保存
//                                                                               其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(DropOutGenMask) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入DropOutGenMask对应空间内并用output_map_指针保存
REG_ADPT_DESC(DropOutGenMask, prim::kPrimDropoutGenMask->name(), ADPT_DESC(DropOutGenMask))
//构造指向DropOutGenMask的指针并储存，创建结构体RegAdptDescDropOutGenMask

// LinSpace
INPUT_MAP(LinSpace) = {{1, INPUT_DESC(start)}, {2, INPUT_DESC(stop)}, {3, INPUT_DESC(num)}};
//将变量start、stop、num处理并存入对应InputDesc结构体的相应变量中，存入LinSpace对应空间内并用input_map指针保存
ATTR_MAP(LinSpace) = EMPTY_ATTR_MAP;//将空变量存入LinSpace对应空间并用attr_map_指针保存
OUTPUT_MAP(LinSpace) = {{0, OUTPUT_DESC(output)}};//将变量output处理并存入对应OutputDesc结构体的相应变量中，存入LinSpace对应空间内并用output_map_指针保存
REG_ADPT_DESC(LinSpace, kNameLinSpace, ADPT_DESC(LinSpace))//构造指向LinSpace的指针并储存，创建结构体RegAdptDescLinSpace

// RandomChoiceWithMask
INPUT_MAP(RandomChoiceWithMask) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入RandomChoiceWithMask对应空间内并用input_map指针保存
ATTR_MAP(RandomChoiceWithMask) = {{"count", ATTR_DESC(count, AnyTraits<int64_t>())},
                                  {"seed", ATTR_DESC(seed, AnyTraits<int64_t>())},
                                  {"seed2", ATTR_DESC(seed2, AnyTraits<int64_t>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                     存入RandomChoiceWithMask对应空间并用attr_map_指针保存
//                                                                                     其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(RandomChoiceWithMask) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(mask)}};
//将变量y、mask处理并存入对应OutputDesc结构体的相应变量中，存入RandomChoiceWithMask对应空间内并用output_map_指针保存
REG_ADPT_DESC(RandomChoiceWithMask, kNameRandomChoiceWithMask, ADPT_DESC(RandomChoiceWithMask))
//构造指向RandomChoiceWithMask的指针并储存，创建结构体RegAdptDescRandomChoiceWithMask

// TruncatedNormal
INPUT_MAP(TruncatedNormal) = {{1, INPUT_DESC(shape)}};//将变量shape处理并存入对应InputDesc结构体的相应变量中，存入TruncatedNormal对应空间内并用input_map指针保存
ATTR_MAP(TruncatedNormal) = {{"seed", ATTR_DESC(seed, AnyTraits<int64_t>())},
                             {"seed2", ATTR_DESC(seed2, AnyTraits<int64_t>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                存入TruncatedNormal对应空间并用attr_map_指针保存
//                                                                                其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(TruncatedNormal) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入TruncatedNormal对应空间内并用output_map_指针保存
REG_ADPT_DESC(TruncatedNormal, kNameTruncatedNormal, ADPT_DESC(TruncatedNormal))
//构造指向TruncatedNormal的指针并储存，创建结构体RegAdptDescTruncatedNormal
}  // namespace mindspore::transform
