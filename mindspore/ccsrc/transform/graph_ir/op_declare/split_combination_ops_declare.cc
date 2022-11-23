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

#include "transform/graph_ir/op_declare/split_combination_ops_declare.h"//按照路径寻找以下文件，导入到本文件
#include <vector>//提供vector数组构建函数模版等

namespace mindspore::transform {//创建名为transform的空间，其空间处于空间mindspore下
// SplitD
INPUT_MAP(SplitD) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入SplitD对应空间内并用input_map指针保存
ATTR_MAP(SplitD) = {{"axis", ATTR_DESC(split_dim, AnyTraits<int64_t>())},
                    {"output_num", ATTR_DESC(num_split, AnyTraits<int64_t>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                存入SplitD对应空间并用attr_map_指针保存
//                                                                                其中AnyTraits<>的作用为将<>内类型进行构建
DYN_OUTPUT_MAP(SplitD) = {{0, DYN_OUTPUT_DESC(y)}};//将变量x处理并存入对应DynOutputDesc结构体的相应变量中，存入SplitD对应空间内并用input_map指针保存
REG_ADPT_DESC(SplitD, kNameSplitD, ADPT_DESC(SplitD))//构造指向SplitD的指针并储存，创建结构体RegAdptDescSplitD

// Pack
INPUT_MAP(Pack) = EMPTY_INPUT_MAP;//将空变量存入Pack对应空间并用input_map指针保存
DYN_INPUT_MAP(Pack) = {{1, DYN_INPUT_DESC(x)}};//将变量x处理并存入对应DynOutputDesc结构体的相应变量中，存入Pack对应空间内并用dyn_input_map_指针保存
ATTR_MAP(Pack) = {{"num", ATTR_DESC(N, AnyTraits<int64_t>())}, {"axis", ATTR_DESC(axis, AnyTraits<int64_t>())}};
//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//存入Pack对应空间并用attr_map_指针保存
//其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(Pack) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入Pack对应空间内并用output_map_指针保存
REG_ADPT_DESC(Pack, prim::kStack, ADPT_DESC(Pack))//构造指向Pack的指针并储存，创建结构体RegAdptDescPack

// ParallelConcat
INPUT_MAP(ParallelConcat) = EMPTY_INPUT_MAP;//将空变量存入ParallelConcat对应空间并用input_map指针保存
DYN_INPUT_MAP(ParallelConcat) = {{1, DYN_INPUT_DESC(values)}};
//将变量values处理并存入对应DynOutputDesc结构体的相应变量中，存入ParallelConcat对应空间内并用dyn_input_map_指针保存
ATTR_MAP(ParallelConcat) = {
  {"shape", ATTR_DESC(shape, AnyTraits<std::vector<int64_t>>())},
  {"N", ATTR_DESC(N, AnyTraits<int64_t>())},
};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//存入ParallelConcat对应空间并用attr_map_指针保存
//其中AnyTraits<>的作用为将<>内类型进行构建
//std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(ParallelConcat) = {{0, OUTPUT_DESC(output_data)}};
//将变量output_data处理并存入对应OutputDesc结构体的相应变量中，存入ParallelConcat对应空间内并用output_map_指针保存
REG_ADPT_DESC(ParallelConcat, kNameParallelConcat, ADPT_DESC(ParallelConcat))
//构造指向ParallelConcat的指针并储存，创建结构体RegAdptDescParallelConcat

// ConcatD
INPUT_MAP(ConcatD) = EMPTY_INPUT_MAP;//将空变量存入ConcatD对应空间并用input_map指针保存
DYN_INPUT_MAP(ConcatD) = {{1, DYN_INPUT_DESC(x)}};//将变量x处理并存入对应DynOutputDesc结构体的相应变量中，存入ConcatD对应空间内并用dyn_input_map_指针保存
ATTR_MAP(ConcatD) = {
  {"axis", ATTR_DESC(concat_dim, AnyTraits<int64_t>())},
  {"inputNums", ATTR_DESC(N, AnyTraits<int64_t>())},
};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//存入ConcatD对应空间并用attr_map_指针保存
//其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(ConcatD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入ConcatD)对应空间内并用output_map_指针保存
REG_ADPT_DESC(ConcatD, prim::kPrimConcat->name(), ADPT_DESC(ConcatD))//构造指向ConcatD的指针并储存，创建结构体RegAdptDescConcatD

// ConcatV2D Inference for tf
INPUT_MAP(ConcatV2D) = EMPTY_INPUT_MAP;//将空变量存入ConcatV2D对应空间并用input_map指针保存
DYN_INPUT_MAP(ConcatV2D) = {{1, DYN_INPUT_DESC(x)}};//将变量x处理并存入对应DynOutputDesc结构体的相应变量中，存入ConcatV2D对应空间内并用dyn_input_map_指针保存
ATTR_MAP(ConcatV2D) = {
  {"axis", ATTR_DESC(concat_dim, AnyTraits<int64_t>())},
  {"N", ATTR_DESC(N, AnyTraits<int64_t>())},
};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//存入ConcaV2tD对应空间并用attr_map_指针保存
//其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(ConcatV2D) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入ConcatV2D对应空间内并用output_map_指针保存
REG_ADPT_DESC(ConcatV2D, kNameConcatV2D, ADPT_DESC(ConcatV2D))//构造指向ConcatV2D的指针并储存，创建结构体RegAdptDescConcatV2D
}  // namespace mindspore::transform
