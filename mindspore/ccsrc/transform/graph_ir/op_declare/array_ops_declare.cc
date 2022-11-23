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

#include "transform/graph_ir/op_declare/array_ops_declare.h"
#include <vector>

namespace mindspore::transform {//创建名为transform的空间，使其位于mindspore空间下
// const
//部分语句与下面语句块作用类似，此处为对变量const进行调整
INPUT_MAP(Const) = EMPTY_INPUT_MAP;//将const与标准进行比较，使原以input_map_为的key变为空
ATTR_MAP(Const) = {{"value", ATTR_DESC(value, AnyTraits<AnyValue>())}};/*
    将value处理并存入对应ATTR_DESC结构体的相应变量中
  将value内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为ATTR_DESC结构体
//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//存入MaxPool对应空间并用attr_map_指针保存
//其中AnyTraits<>的作用为将<>内类型进行构建
将Const的类型与标准进行对比，后进行空间调整，并用指针attr_map_为key存储相应内容
*/
OUTPUT_MAP(Const) = {{0, OUTPUT_DESC(y)}};//用新建的OUTPUT结构体并与标准比较其容量，以output_map_为变量的key

// Constant
//此处为对变量Constant进行调整
INPUT_MAP(Constant) = EMPTY_INPUT_MAP;
ATTR_MAP(Constant) = {{"value", ATTR_DESC(value, AnyTraits<AnyValue>())}};
OUTPUT_MAP(Constant) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Constant, kNameConst, ADPT_DESC(Constant, Const))//将Constant处理并存入对应REG_ADPT_DESC结构体的相应变量中
  //将Constant内容转为字符串变量并存储至结构体的name变量中
  //引用Operator空间并将指针所指的类转为结构体

// ScalarSummary
INPUT_MAP(Summary) = {{2, INPUT_DESC(x)}};
ATTR_MAP(Summary) = EMPTY_ATTR_MAP;
#ifndef ENABLE_SECURITY
REG_ADPT_DESC(ScalarSummary, prim::kPrimScalarSummary->name(), ADPT_DESC(Summary))
REG_ADPT_DESC(ImageSummary, prim::kPrimImageSummary->name(), ADPT_DESC(Summary))
REG_ADPT_DESC(TensorSummary, prim::kPrimTensorSummary->name(), ADPT_DESC(Summary))
REG_ADPT_DESC(HistogramSummary, prim::kPrimHistogramSummary->name(), ADPT_DESC(Summary))
#endif
REG_ADPT_DESC(Debug, prim::kPrimDebug->name(), ADPT_DESC(Summary))

// Data
INPUT_MAP(Data) = EMPTY_INPUT_MAP;
ATTR_MAP(Data) = EMPTY_ATTR_MAP;
REG_ADPT_DESC(Data, kNameParam, ADPT_DESC(Data))

// Shape
INPUT_MAP(Shape) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Shape) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Shape) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Shape, kNameShape, ADPT_DESC(Shape))

// Reshape
INPUT_MAP(Reshape) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(shape)}};
ATTR_MAP(Reshape) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Reshape) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Reshape, kNameReshape, ADPT_DESC(Reshape))
REG_ADPT_DESC(FlattenGrad, kNameFlattenGrad, ADPT_DESC(Reshape))

// TransShape
INPUT_MAP(TransShape) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(TransShape) = {{2, ATTR_DESC(outShape, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(TransShape) = EMPTY_ATTR_MAP;
OUTPUT_MAP(TransShape) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(TransShape, kNameTransShape, ADPT_DESC(TransShape))

// MirrorPad
INPUT_MAP(MirrorPad) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(paddings)}};
ATTR_MAP(MirrorPad) = {{"mode", ATTR_DESC(mode, AnyTraits<std::string>())}};
OUTPUT_MAP(MirrorPad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MirrorPad, kNameMirrorPad, ADPT_DESC(MirrorPad))

// MirrorPadGrad
INPUT_MAP(MirrorPadGrad) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(paddings)}};
ATTR_MAP(MirrorPadGrad) = {{"mode", ATTR_DESC(mode, AnyTraits<std::string>())}};
OUTPUT_MAP(MirrorPadGrad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MirrorPadGrad, kNameMirrorPadGrad, ADPT_DESC(MirrorPadGrad))

// ExpandDims
INPUT_MAP(ExpandDims) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(axis)}};
ATTR_MAP(ExpandDims) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ExpandDims) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ExpandDims, kNameExpandDims, ADPT_DESC(ExpandDims))

// Squeeze
INPUT_MAP(Squeeze) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Squeeze) = {{"axis", ATTR_DESC(axis, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(Squeeze) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Squeeze, prim::kPrimSqueeze->name(), ADPT_DESC(Squeeze))

// ReverseSequence
INPUT_MAP(ReverseSequence) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(seq_lengths)}};
ATTR_MAP(ReverseSequence) = {{"seq_dim", ATTR_DESC(seq_dim, AnyTraits<int64_t>())},
                             {"batch_dim", ATTR_DESC(batch_dim, AnyTraits<int64_t>())}};
OUTPUT_MAP(ReverseSequence) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ReverseSequence, kNameReverseSequence, ADPT_DESC(ReverseSequence))

// EditDistance
INPUT_MAP(EditDistance) = {{1, INPUT_DESC(hypothesis_indices)}, {2, INPUT_DESC(hypothesis_values)},
                           {3, INPUT_DESC(hypothesis_shape)},   {4, INPUT_DESC(truth_indices)},
                           {5, INPUT_DESC(truth_values)},       {6, INPUT_DESC(truth_shape)}};
ATTR_MAP(EditDistance) = {{"normalize", ATTR_DESC(normalize, AnyTraits<bool>())}};
OUTPUT_MAP(EditDistance) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(EditDistance, kNameEditDistance, ADPT_DESC(EditDistance))

// NonZero
INPUT_MAP(NonZero) = {{1, INPUT_DESC(x)}};
ATTR_MAP(NonZero) = {{"transpose", ATTR_DESC(transpose, AnyTraits<bool>())}};
OUTPUT_MAP(NonZero) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(NonZero, kNameNonZero, ADPT_DESC(NonZero))

// Unsqueeze
INPUT_MAP(Unsqueeze) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Unsqueeze) = {{"axis", ATTR_DESC(axes, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(Unsqueeze) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Unsqueeze, kNameUnsqueeze, ADPT_DESC(Unsqueeze))
}  // namespace mindspore::transform
