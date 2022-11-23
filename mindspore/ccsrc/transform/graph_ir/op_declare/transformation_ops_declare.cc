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

#include "transform/graph_ir/op_declare/transformation_ops_declare.h"//按照路径寻找以下文件，导入到本文件
#include <vector>//提供vector数组构建函数模版等

namespace mindspore::transform {//创建名为transform的空间，其空间处于空间mindspore下
// Flatten
INPUT_MAP(Flatten) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入Flatten对应空间内并用input_map指针保存
ATTR_MAP(Flatten) = EMPTY_ATTR_MAP;//将空变量存入Flatten对应空间并用attr_map_指针保存
OUTPUT_MAP(Flatten) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入Flatten对应空间内并用output_map_指针保存
REG_ADPT_DESC(Flatten, prim::kPrimFlatten->name(), ADPT_DESC(Flatten))//构造指向Flatten的指针并储存，创建结构体RegAdptDescFlatten

// Unpack
INPUT_MAP(Unpack) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入Unpack对应空间内并用input_map指针保存
ATTR_MAP(Unpack) = {{"axis", ATTR_DESC(axis, AnyTraits<int64_t>())}, {"num", ATTR_DESC(num, AnyTraits<int64_t>())}};
//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//存入Unpack对应空间并用attr_map_指针保存
//其中AnyTraits<>的作用为将<>内类型进行构建
DYN_OUTPUT_MAP(Unpack) = {{0, DYN_OUTPUT_DESC(y)}};//将变量y处理并存入对应DynOutputDesc结构体的相应变量中，存入Unpack对应空间内并用input_map指针保存
REG_ADPT_DESC(Unpack, prim::kUnstack, ADPT_DESC(Unpack))//构造指向Unpack的指针并储存，创建结构体RegAdptDescUnpack

// ExtractImagePatches
INPUT_MAP(ExtractImagePatches) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入ExtractImagePatches对应空间内并用input_map指针保存
ATTR_MAP(ExtractImagePatches) = {
  {"ksizes", ATTR_DESC(ksizes, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"strides", ATTR_DESC(strides, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"rates", ATTR_DESC(rates, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
  {"padding", ATTR_DESC(padding, AnyTraits<std::string>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                             存入ParallelConcat对应空间并用attr_map_指针保存
//                                                             其中AnyTraits<>的作用为将<>内类型进行构建
//                                                             std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(ExtractImagePatches) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入ExtractImagePatches对应空间内并用output_map_指针保存
REG_ADPT_DESC(ExtractImagePatches, kNameExtractImagePatches, ADPT_DESC(ExtractImagePatches))
//构造指向ExtractImagePatches的指针并储存，创建结构体RegAdptDescExtractImagePatches

// Transpose
INPUT_MAP(TransposeD) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入TransposeD对应空间内并用input_map指针保存
INPUT_ATTR_MAP(TransposeD) = {{2, ATTR_DESC(perm, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                                             存入TransposeD对应空间并用attr_map_指针保存
//                                                                                                             其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                                                             std:vector<>的作用为构建<>内类型的容量可变的数组
ATTR_MAP(TransposeD) = EMPTY_ATTR_MAP;//将空变量存入TransposeD对应空间并用attr_map_指针保存
// Do not set Transpose operator output descriptor不要在输出描述符中设置转置运算符
REG_ADPT_DESC(TransposeD, prim::kPrimTranspose->name(), ADPT_DESC(TransposeD))
//构造指向TransposeD的指针并储存，创建结构体RegAdptDescTransposeD

// SpaceToDepth
INPUT_MAP(SpaceToDepth) = {{1, INPUT_DESC(x)}};;//将变量x处理并存入对应InputDesc结构体的相应变量中，存入SpaceToDepth对应空间内并用input_map指针保存
ATTR_MAP(SpaceToDepth) = {{"block_size", ATTR_DESC(block_size, AnyTraits<int64_t>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                      存入SpaceToDepth对应空间并用attr_map_指针保存
//                                                                                      其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(SpaceToDepth) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入SpaceToDepth对应空间内并用output_map_指针保存
REG_ADPT_DESC(SpaceToDepth, kNameSpaceToDepth, ADPT_DESC(SpaceToDepth))//构造指向SpaceToDepth的指针并储存，创建结构体RegAdptDescSpaceToDepth

// DepthToSpace
INPUT_MAP(DepthToSpace) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入DepthToSpace对应空间内并用input_map指针保存
ATTR_MAP(DepthToSpace) = {{"block_size", ATTR_DESC(block_size, AnyTraits<int64_t>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                       存入DepthToSpace对应空间并用attr_map_指针保存
//                                                                                       其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(DepthToSpace) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入DepthToSpace对应空间内并用output_map_指针保存
REG_ADPT_DESC(DepthToSpace, kNameDepthToSpace, ADPT_DESC(DepthToSpace))//构造指向DepthToSpace的指针并储存，创建结构体RegAdptDescDepthToSpace

// SpaceToBatchD
INPUT_MAP(SpaceToBatchD) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入SpaceToBatchD对应空间内并用input_map指针保存
ATTR_MAP(SpaceToBatchD) = {
  {"block_size", ATTR_DESC(block_size, AnyTraits<int64_t>())},
  {"paddings", ATTR_DESC(paddings, AnyTraits<std::vector<std::vector<int64_t>>>(), AnyTraits<std::vector<int64_t>>())}};
//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//存入SpaceToBatchD对应空间并用attr_map_指针保存
//其中AnyTraits<>的作用为将<>内类型进行构建
//std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(SpaceToBatchD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入SpaceToBatchD对应空间内并用output_map_指针保存
REG_ADPT_DESC(SpaceToBatchD, kNameSpaceToBatch, ADPT_DESC(SpaceToBatchD))//构造指向SpaceToBatchD的指针并储存，创建结构体RegAdptDescSpaceToBatchD

// SpaceToBatchNDD
INPUT_MAP(SpaceToBatchNDD) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入SpaceToBatchNDD对应空间内并用input_map指针保存
ATTR_MAP(SpaceToBatchNDD) = {
  {"block_shape", ATTR_DESC(block_shape, AnyTraits<std::vector<int64_t>>())},
  {"paddings", ATTR_DESC(paddings, AnyTraits<std::vector<std::vector<int64_t>>>(), AnyTraits<std::vector<int64_t>>())}};
//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//存入SpaceToBatchNDD对应空间并用attr_map_指针保存
//其中AnyTraits<>的作用为将<>内类型进行构建
//std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(SpaceToBatchNDD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入SpaceToBatchNDD对应空间内并用output_map_指针保存
REG_ADPT_DESC(SpaceToBatchNDD, kNameSpaceToBatchNDD, ADPT_DESC(SpaceToBatchNDD))//构造指向SpaceToBatchNDD的指针并储存，创建结构体RegAdptDescSpaceToBatchNDD

// BatchToSpaceD
INPUT_MAP(BatchToSpaceD) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入BatchToSpaceD对应空间内并用input_map指针保存
ATTR_MAP(BatchToSpaceD) = {
  {"block_size", ATTR_DESC(block_size, AnyTraits<int64_t>())},
  {"crops", ATTR_DESC(crops, AnyTraits<std::vector<std::vector<int64_t>>>(), AnyTraits<std::vector<int64_t>>())}};
//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//存入BatchToSpaceD对应空间并用attr_map_指针保存
//其中AnyTraits<>的作用为将<>内类型进行构建
//std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(BatchToSpaceD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入BatchToSpaceD对应空间内并用output_map_指针保存
REG_ADPT_DESC(BatchToSpaceD, kNameBatchToSpace, ADPT_DESC(BatchToSpaceD))//构造指向BatchToSpaceD的指针并储存，创建结构体RegAdptDescBatchToSpaceD

// BatchToSpaceNDD
INPUT_MAP(BatchToSpaceNDD) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入BatchToSpaceNDD对应空间内并用input_map指针保存
ATTR_MAP(BatchToSpaceNDD) = {
  {"block_shape", ATTR_DESC(block_shape, AnyTraits<std::vector<int64_t>>())},
  {"crops", ATTR_DESC(crops, AnyTraits<std::vector<std::vector<int64_t>>>(), AnyTraits<std::vector<int64_t>>())}};
//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//存入BatchToSpaceNDD对应空间并用attr_map_指针保存
//其中AnyTraits<>的作用为将<>内类型进行构建
//std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(BatchToSpaceNDD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入BatchToSpaceNDD对应空间内并用output_map_指针保存
REG_ADPT_DESC(BatchToSpaceNDD, kNameBatchToSpaceNd, ADPT_DESC(BatchToSpaceNDD))//构造指向BatchToSpaceNDD的指针并储存，创建结构体RegAdptDescBatchToSpaceNDD
}  // namespace mindspore::transform
