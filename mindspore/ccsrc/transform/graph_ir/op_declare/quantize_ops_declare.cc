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

#include "transform/graph_ir/op_declare/quantize_ops_declare.h"//按照路径寻找以下文件，导入到本文件

namespace mindspore::transform {//创建名为transform的空间，其空间处于空间mindspore下
// AscendQuant
INPUT_MAP(AscendQuant) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入AscendQuant对应空间内并用input_map指针保存
ATTR_MAP(AscendQuant) = {{"scale", ATTR_DESC(scale, AnyTraits<float>())},
                         {"offset", ATTR_DESC(offset, AnyTraits<float>())},
                         {"sqrt_mode", ATTR_DESC(sqrt_mode, AnyTraits<bool>())},
                         {"round_mode", ATTR_DESC(round_mode, AnyTraits<std::string>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                          存入AscendQuant对应空间并用attr_map_指针保存
//                                                                                          其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                                          std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(AscendQuant) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入AscendQuant对应空间内并用output_map_指针保存
REG_ADPT_DESC(AscendQuant, kNameAscendQuant, ADPT_DESC(AscendQuant))//构造指向AscendQuant的指针并储存，创建结构体RegAdptDescAscendQuant

// AscendDequant
INPUT_MAP(AscendDequant) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(deq_scale)}};
//将变量x、deq_scale处理并存入对应InputDesc结构体的相应变量中，存入AscendDequant对应空间内并用input_map指针保存
ATTR_MAP(AscendDequant) = {{"sqrt_mode", ATTR_DESC(sqrt_mode, AnyTraits<bool>())},
                           {"relu_flag", ATTR_DESC(relu_flag, AnyTraits<bool>())},
                           {"dtype", ATTR_DESC(dtype, AnyTraits<GEType>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                             存入AscendDequant对应空间并用attr_map_指针保存
//                                                                             其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                             std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(AscendDequant) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入AscendDequant对应空间内并用output_map_指针保存
REG_ADPT_DESC(AscendDequant, kNameAscendDequant, ADPT_DESC(AscendDequant))//构造指向AscendDequant的指针并储存，创建结构体RegAdptDescAscendDequant
}  // namespace mindspore::transform
