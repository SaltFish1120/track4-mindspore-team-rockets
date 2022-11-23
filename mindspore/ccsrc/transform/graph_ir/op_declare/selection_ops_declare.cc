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

#include <vector>//提供vector数组构建函数模版等
#include "transform/graph_ir/op_declare/selection_ops_declare.h"//按照路径寻找以下文件，导入到本文件

namespace mindspore::transform {//创建名为transform的空间，其空间处于空间mindspore下
// CumsumD
INPUT_MAP(CumsumD) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入CumsumD对应空间内并用input_map指针保存
INPUT_ATTR_MAP(CumsumD) = {{2, ATTR_DESC(axis, AnyTraits<int64_t>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                       存入CumsumD对应空间并用attr_map_指针保存
//                                                                       其中AnyTraits<>的作用为将<>内类型进行构建
ATTR_MAP(CumsumD) = {{"exclusive", ATTR_DESC(exclusive, AnyTraits<bool>())},
                     {"reverse", ATTR_DESC(reverse, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                         存入CumsumD对应空间并用attr_map_指针保存
//                                                                         其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(CumsumD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入CumsumD对应空间内并用output_map_指针保存
REG_ADPT_DESC(CumsumD, kNameCumSum, ADPT_DESC(CumsumD))//构造指向CumsumD的指针并储存，创建结构体RegAdptDescCumsumD

// GatherV2
INPUT_MAP(GatherV2) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(axis)}};
//将变量x、indices、axis处理并存入对应InputDesc结构体的相应变量中，存入GatherV2对应空间内并用input_map指针保存
ATTR_MAP(GatherV2) = EMPTY_ATTR_MAP;//将空变量存入GatherV2对应空间并用attr_map_指针保存
OUTPUT_MAP(GatherV2) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入GatherV2对应空间内并用output_map_指针保存

// CumprodD
INPUT_MAP(CumprodD) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入CumprodD对应空间内并用input_map指针保存
INPUT_ATTR_MAP(CumprodD) = {{2, ATTR_DESC(axis, AnyTraits<int64_t>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                        存入CumprodD对应空间并用attr_map_指针保存
//                                                                        其中AnyTraits<>的作用为将<>内类型进行构建
ATTR_MAP(CumprodD) = {{"exclusive", ATTR_DESC(exclusive, AnyTraits<bool>())},
                      {"reverse", ATTR_DESC(reverse, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                          存入CumprodD对应空间并用attr_map_指针保存
//                                                                          其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(CumprodD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入CumprodD对应空间内并用output_map_指针保存
REG_ADPT_DESC(CumprodD, kNameCumProd, ADPT_DESC(CumprodD))//构造指向CumprodD的指针并储存，创建结构体RegAdptDescCumprodD

//SliceD
INPUT_MAP(SliceD) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入SliceD对应空间内并用input_map指针保存
INPUT_ATTR_MAP(SliceD) = {{2, ATTR_DESC(offsets, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                          {3, ATTR_DESC(size, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                                         存入SliceD对应空间并用attr_map_指针保存
//                                                                                                         其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                                                          std:vector<>的作用为构建<>内类型的容量可变的数组
ATTR_MAP(SliceD) = EMPTY_ATTR_MAP;//将空变量存入SliceD对应空间并用attr_map_指针保存
OUTPUT_MAP(SliceD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入SliceD对应空间内并用output_map_指针保存
REG_ADPT_DESC(SliceD, kNameSlice, ADPT_DESC(SliceD))//构造指向SliceD的指针并储存，创建结构体RegAdptDescSliceD

// TopK
INPUT_MAP(TopK) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(k)}};
//将变量x、k处理并存入对应InputDesc结构体的相应变量中，存入TopK对应空间内并用input_map指针保存
ATTR_MAP(TopK) = {{"sorted", ATTR_DESC(sorted, AnyTraits<bool>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                    存入TopK对应空间并用attr_map_指针保存
//                                                                    其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(TopK) = {{0, OUTPUT_DESC(values)}, {1, OUTPUT_DESC(indices)}};
//将变量values、indices处理并存入对应OutputDesc结构体的相应变量中，存入TopK对应空间内并用output_map_指针保存
REG_ADPT_DESC(TopK, kNameTopK, ADPT_DESC(TopK))//构造指向TopK的指针并储存，创建结构体RegAdptDescTopK

// InTopK
INPUT_MAP(InTopKD) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};//将变量x1、x2处理并存入对应InputDesc结构体的相应变量中，存入InTopKD对应空间内并用input_map指针保存
ATTR_MAP(InTopKD) = {{"k", ATTR_DESC(k, AnyTraits<int64_t>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                存入InTopKD对应空间并用attr_map_指针保存
//                                                                其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(InTopKD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入InTopKD对应空间内并用output_map_指针保存
REG_ADPT_DESC(InTopKD, kNameInTopKD, ADPT_DESC(InTopKD))//构造指向InTopKD的指针并储存，创建结构体RegAdptDescInTopKD

// TileD
INPUT_MAP(TileD) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入TileD对应空间内并用input_map指针保存
INPUT_ATTR_MAP(TileD) = {{2, ATTR_DESC(multiples, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                                         存入SliceD对应空间并用attr_map_指针保存
//                                                                                                         其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                                                          std:vector<>的作用为构建<>内类型的容量可变的数组
ATTR_MAP(TileD) = EMPTY_ATTR_MAP;//将空变量存入TileD对应空间并用attr_map_指针保存
OUTPUT_MAP(TileD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入TileD对应空间内并用output_map_指针保存
REG_ADPT_DESC(TileD, kNameTile, ADPT_DESC(TileD))//构造指向TileD的指针并储存，创建结构体RegAdptDescTileD

// OneHot
INPUT_MAP(OneHot) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(depth)}, {3, INPUT_DESC(on_value)}, {4, INPUT_DESC(off_value)}};
//将变量x、depth、on_value、off_value处理并存入对应InputDesc结构体的相应变量中，存入OneHot对应空间内并用input_map指针保存
ATTR_MAP(OneHot) = {{"axis", ATTR_DESC(axis, AnyTraits<int64_t>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                    存入OneHot对应空间并用attr_map_指针保存
//                                                                    其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(OneHot) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入OneHot对应空间内并用output_map_指针保存
REG_ADPT_DESC(OneHot, prim::kPrimOneHot->name(), ADPT_DESC(OneHot))//构造指向OneHot的指针并储存，创建结构体RegAdptDescOneHot

// GatherV2D
INPUT_MAP(GatherV2D) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(indices)}};
//将变量x、indices处理并存入对应InputDesc结构体的相应变量中，存入GatherV2D对应空间内并用input_map指针保存
INPUT_ATTR_MAP(GatherV2D) = {{3, ATTR_DESC(axis, AnyTraits<int64_t>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                         存入GatherV2D对应空间并用attr_map_指针保存
//                                                                         其中AnyTraits<>的作用为将<>内类型进行构建
ATTR_MAP(GatherV2D) = EMPTY_ATTR_MAP;//将空变量存入GatherV2D对应空间并用attr_map_指针保存
OUTPUT_MAP(GatherV2D) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入GatherV2D对应空间内并用output_map_指针保存
REG_ADPT_DESC(GatherV2D, prim::kPrimGather->name(), ADPT_DESC(GatherV2D))//构造指向GatherV2D的指针并储存，创建结构体RegAdptDescGatherV2D
REG_ADPT_DESC(Gather, kNameGather, ADPT_DESC(GatherV2D))//构造指向Gather的指针并储存，创建结构体RegAdptDescGather

// ScatterNdD
INPUT_MAP(ScatterNdD) = {{1, INPUT_DESC(indices)}, {2, INPUT_DESC(x)}};
//将变量indices、x处理并存入对应InputDesc结构体的相应变量中，存入ScatterNdD对应空间内并用input_map指针保存
INPUT_ATTR_MAP(ScatterNdD) = {
  {3, ATTR_DESC(shape, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                               存入ScatterNdD对应空间并用attr_map_指针保存
//                                                                                               其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                                               std:vector<>的作用为构建<>内类型的容量可变的数组
ATTR_MAP(ScatterNdD) = EMPTY_ATTR_MAP;//将空变量存入ScatterNdD对应空间并用attr_map_指针保存
OUTPUT_MAP(ScatterNdD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入ScatterNdD对应空间内并用output_map_指针保存
REG_ADPT_DESC(ScatterNdD, kNameScatterNdD, ADPT_DESC(ScatterNdD))//构造指向ScatterNdD的指针并储存，创建结构体RegAdptDescScatterNdD

// ScatterNonAliasingAdd
INPUT_MAP(ScatterNonAliasingAdd) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(updates)}};
//将变量x、indices、updates处理并存入对应InputDesc结构体的相应变量中，存入ScatterNonAliasingAdd对应空间内并用input_map指针保存
ATTR_MAP(ScatterNonAliasingAdd) = EMPTY_ATTR_MAP;//将空变量存入ScatterNonAliasingAdd对应空间并用attr_map_指针保存
OUTPUT_MAP(ScatterNonAliasingAdd) = {{0, OUTPUT_DESC(y)}};
//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入ScatterNonAliasingAdd对应空间内并用output_map_指针保存
REG_ADPT_DESC(ScatterNonAliasingAdd, kNameScatterNonAliasingAdd, ADPT_DESC(ScatterNonAliasingAdd))
//构造指向ScatterNonAliasingAdd的指针并储存，创建结构体RegAdptDescScatterNonAliasingAdd

// GatherNd
INPUT_MAP(GatherNd) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(indices)}};
//将变量x、indices处理并存入对应InputDesc结构体的相应变量中，存入GatherNd对应空间内并用input_map指针保存
ATTR_MAP(GatherNd) = EMPTY_ATTR_MAP;//将空变量存入GatherNd对应空间并用attr_map_指针保存
OUTPUT_MAP(GatherNd) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入GatherNd对应空间内并用output_map_指针保存
REG_ADPT_DESC(GatherNd, kNameGatherNd, ADPT_DESC(GatherNd))//构造指向GatherNd的指针并储存，创建结构体RegAdptDescGatherNd

// GatherD
INPUT_MAP(GatherD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(dim)}, {3, INPUT_DESC(index)}};
//将变量x、dim、index处理并存入对应InputDesc结构体的相应变量中，存入GatherD对应空间内并用input_map指针保存
ATTR_MAP(GatherD) = EMPTY_ATTR_MAP;//将空变量存入GatherD对应空间并用attr_map_指针保存
OUTPUT_MAP(GatherD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入GatherD对应空间内并用output_map_指针保存
REG_ADPT_DESC(GatherD, kNameGatherD, ADPT_DESC(GatherD))//构造指向GatherD的指针并储存，创建结构体RegAdptDescGatherD

// Range
INPUT_MAP(RangeD) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入RangeD对应空间内并用input_map指针保存
ATTR_MAP(RangeD) = {{"start", ATTR_DESC(start, AnyTraits<float>())},
                    {"limit", ATTR_DESC(limit, AnyTraits<float>())},
                    {"delta", ATTR_DESC(delta, AnyTraits<float>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                     存入RangeD对应空间并用attr_map_指针保存
//                                                                     其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(RangeD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入RangeD对应空间内并用output_map_指针保存
REG_ADPT_DESC(RangeD, kNameRange, ADPT_DESC(RangeD))//构造指向RangeD的指针并储存，创建结构体RegAdptDescRangeD

// InplaceAddD
INPUT_MAP(InplaceAddD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(v)}};
//将变量x、v处理并存入对应InputDesc结构体的相应变量中，存入InplaceAddD对应空间内并用input_map指针保存
ATTR_MAP(InplaceAddD) = {{"indices", ATTR_DESC(indices, AnyTraits<std::vector<int64_t>>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                             存入InplaceAddD对应空间并用attr_map_指针保存
//                                                                                             其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                                             std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(InplaceAddD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入InplaceAddD对应空间内并用output_map_指针保存
REG_ADPT_DESC(InplaceAddD, kNameInplaceAddD, ADPT_DESC(InplaceAddD))//构造指向InplaceAddD的指针并储存，创建结构体RegAdptDescInplaceAddD

// InplaceSubD
INPUT_MAP(InplaceSubD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(v)}};
//将变量x、v处理并存入对应InputDesc结构体的相应变量中，存入InplaceSubD对应空间内并用input_map指针保存
ATTR_MAP(InplaceSubD) = {{"indices", ATTR_DESC(indices, AnyTraits<std::vector<int64_t>>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                             存入InplaceSubD对应空间并用attr_map_指针保存
//                                                                                             其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                                             std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(InplaceSubD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入InplaceSubD对应空间内并用output_map_指针保存
REG_ADPT_DESC(InplaceSubD, kNameInplaceSubD, ADPT_DESC(InplaceSubD))//构造指向InplaceSubD的指针并储存，创建结构体RegAdptDescInplaceSubD

// InplaceUpdateD
INPUT_MAP(InplaceUpdateD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(v)}};
//将变量x、v处理并存入对应InputDesc结构体的相应变量中，存入InplaceUpdateD对应空间内并用input_map指针保存
ATTR_MAP(InplaceUpdateD) = {{"indices", ATTR_DESC(indices, AnyTraits<std::vector<int64_t>>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                                存入InplaceUpdateD对应空间并用attr_map_指针保存
//                                                                                                其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                                                std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(InplaceUpdateD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入InplaceUpdateD对应空间内并用output_map_指针保存
REG_ADPT_DESC(InplaceUpdateD, kNameInplaceUpdateD, ADPT_DESC(InplaceUpdateD))//构造指向InplaceUpdateD的指针并储存，创建结构体RegAdptDescInplaceUpdateD

// Select
INPUT_MAP(Select) = {{1, INPUT_DESC(condition)}, {2, INPUT_DESC(x1)}, {3, INPUT_DESC(x2)}};
//将变量condition、x1、x2处理并存入对应InputDesc结构体的相应变量中，存入Select对应空间内并用input_map指针保存
ATTR_MAP(Select) = EMPTY_ATTR_MAP;//将空变量存入Select对应空间并用attr_map_指针保存
OUTPUT_MAP(Select) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入InplaceUpdateD对应空间内并用output_map_指针保存
REG_ADPT_DESC(Select, prim::kPrimSelect->name(), ADPT_DESC(Select))//构造指向Select的指针并储存，创建结构体RegAdptDescSelect

// StridedSliceGrad
INPUT_MAP(StridedSliceGrad) = {
  {1, INPUT_DESC(dy)}, {2, INPUT_DESC(shape)}, {3, INPUT_DESC(begin)}, {4, INPUT_DESC(end)}, {5, INPUT_DESC(strides)}};
//将变量dy、shape、begin、end、strides处理并存入对应InputDesc结构体的相应变量中，存入StridedSliceGrad对应空间内并用input_map指针保存
ATTR_MAP(StridedSliceGrad) = {{"begin_mask", ATTR_DESC(begin_mask, AnyTraits<int64_t>())},
                              {"end_mask", ATTR_DESC(end_mask, AnyTraits<int64_t>())},
                              {"ellipsis_mask", ATTR_DESC(ellipsis_mask, AnyTraits<int64_t>())},
                              {"new_axis_mask", ATTR_DESC(new_axis_mask, AnyTraits<int64_t>())},
                              {"shrink_axis_mask", ATTR_DESC(shrink_axis_mask, AnyTraits<int64_t>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                                       存入StridedSliceGrad对应空间并用attr_map_指针保存
//                                                                                                       其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(StridedSliceGrad) = {{0, OUTPUT_DESC(output)}};
//将变量output处理并存入对应OutputDesc结构体的相应变量中，存入StridedSliceGrad对应空间内并用output_map_指针保存
REG_ADPT_DESC(StridedSliceGrad, kNameStridedSliceGrad, ADPT_DESC(StridedSliceGrad))
//构造指向StridedSliceGrad的指针并储存，创建结构体RegAdptDescStridedSliceGrad

// StridedSlice
INPUT_MAP(StridedSlice) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(begin)}, {3, INPUT_DESC(end)}, {4, INPUT_DESC(strides)}};
//将变量x、begin、end、strides处理并存入对应InputDesc结构体的相应变量中，存入StridedSlice对应空间内并用input_map指针保存
ATTR_MAP(StridedSlice) = {{"begin_mask", ATTR_DESC(begin_mask, AnyTraits<int64_t>())},
                          {"end_mask", ATTR_DESC(end_mask, AnyTraits<int64_t>())},
                          {"ellipsis_mask", ATTR_DESC(ellipsis_mask, AnyTraits<int64_t>())},
                          {"new_axis_mask", ATTR_DESC(new_axis_mask, AnyTraits<int64_t>())},
                          {"shrink_axis_mask", ATTR_DESC(shrink_axis_mask, AnyTraits<int64_t>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                                   存入StridedSlice对应空间并用attr_map_指针保存
//                                                                                                    其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(StridedSlice) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入StridedSlice对应空间内并用output_map_指针保存
REG_ADPT_DESC(StridedSlice, kNameStridedSlice, ADPT_DESC(StridedSlice))
//构造指向StridedSlice的指针并储存，创建结构体RegAdptDescStridedSlice

// StridedSliceV2
INPUT_MAP(StridedSliceV2) = {
  {1, INPUT_DESC(x)}, {2, INPUT_DESC(begin)}, {3, INPUT_DESC(end)}, {4, INPUT_DESC(axes)}, {5, INPUT_DESC(strides)}};
//将变量x、begin、end、axes、strides处理并存入对应InputDesc结构体的相应变量中，存入StridedSlice对应空间内并用input_map指针保存
ATTR_MAP(StridedSliceV2) = {{"begin_mask", ATTR_DESC(begin_mask, AnyTraits<int64_t>())},
                            {"end_mask", ATTR_DESC(end_mask, AnyTraits<int64_t>())},
                            {"ellipsis_mask", ATTR_DESC(ellipsis_mask, AnyTraits<int64_t>())},
                            {"new_axis_mask", ATTR_DESC(new_axis_mask, AnyTraits<int64_t>())},
                            {"shrink_axis_mask", ATTR_DESC(shrink_axis_mask, AnyTraits<int64_t>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                                    存入StridedSliceV2对应空间并用attr_map_指针保存
//                                                                                                    其中AnyTraits<>的作用为将<>内类型进行构建
OUTPUT_MAP(StridedSliceV2) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入StridedSliceV2对应空间内并用output_map_指针保存
REG_ADPT_DESC(StridedSliceV2, kNameStridedSliceV2, ADPT_DESC(StridedSliceV2))
//构造指向StridedSliceV2的指针并储存，创建结构体RegAdptDescStridedSliceV2

// UnsortedSegmentSum
INPUT_MAP(UnsortedSegmentSumD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(segment_ids)}};
//将变量x、segment_ids处理并存入对应InputDesc结构体的相应变量中，存入UnsortedSegmentSumD对应空间内并用input_map指针保存
INPUT_ATTR_MAP(UnsortedSegmentSumD) = {{3, ATTR_DESC(num_segments, AnyTraits<int64_t>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                           存入UnsortedSegmentSumD对应空间并用attr_map_指针保存
//                                                                                           其中AnyTraits<>的作用为将<>内类型进行构建
ATTR_MAP(UnsortedSegmentSumD) = EMPTY_ATTR_MAP;//将空变量存入UnsortedSegmentSumD对应空间并用attr_map_指针保存
OUTPUT_MAP(UnsortedSegmentSumD) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入UnsortedSegmentSumD对应空间内并用output_map_指针保存
REG_ADPT_DESC(UnsortedSegmentSumD, prim::kPrimUnsortedSegmentSum->name(), ADPT_DESC(UnsortedSegmentSumD))
//构造指向UnsortedSegmentSumD的指针并储存，创建结构体RegAdptDescUnsortedSegmentSumD

// UnsortedSegmentProdD
INPUT_MAP(UnsortedSegmentProdD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(segment_ids)}};
//将变量x、segment_ids处理并存入对应InputDesc结构体的相应变量中，存入UnsortedSegmentProdD对应空间内并用input_map指针保存
INPUT_ATTR_MAP(UnsortedSegmentProdD) = {{3, ATTR_DESC(num_segments, AnyTraits<int64_t>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                           存入UnsortedSegmentProdD对应空间并用attr_map_指针保存
//                                                                                           其中AnyTraits<>的作用为将<>内类型进行构建
ATTR_MAP(UnsortedSegmentProdD) = EMPTY_ATTR_MAP;//将空变量存入UnsortedSegmentProdD对应空间并用attr_map_指针保存
OUTPUT_MAP(UnsortedSegmentProdD) = {{0, OUTPUT_DESC(y)}};
//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入UnsortedSegmentProdD对应空间内并用output_map_指针保存
REG_ADPT_DESC(UnsortedSegmentProdD, kNameUnsortedSegmentProdD, ADPT_DESC(UnsortedSegmentProdD))
//构造指向UnsortedSegmentProdD的指针并储存，创建结构体RegAdptDescUnsortedSegmentProdD

// UnsortedSegmentMaxD
INPUT_MAP(UnsortedSegmentMaxD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(segment_ids)}};
//将变量x、segment_ids处理并存入对应InputDesc结构体的相应变量中，存入UnsortedSegmentMaxD对应空间内并用input_map指针保存
INPUT_ATTR_MAP(UnsortedSegmentMaxD) = {{3, ATTR_DESC(num_segments, AnyTraits<int64_t>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                           存入UnsortedSegmentMaxD对应空间并用attr_map_指针保存
//                                                                                           其中AnyTraits<>的作用为将<>内类型进行构建
ATTR_MAP(UnsortedSegmentMaxD) = EMPTY_ATTR_MAP;//将空变量存入UnsortedSegmentMaxD对应空间并用attr_map_指针保存
OUTPUT_MAP(UnsortedSegmentMaxD) = {{0, OUTPUT_DESC(y)}};
//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入UnsortedSegmentMaxD对应空间内并用output_map_指针保存
REG_ADPT_DESC(UnsortedSegmentMaxD, kNameUnsortedSegmentMaxD, ADPT_DESC(UnsortedSegmentMaxD))
//构造指向UnsortedSegmentMaxD的指针并储存，创建结构体RegAdptDescUnsortedSegmentMaxD

// UnsortedSegmentMin
INPUT_MAP(UnsortedSegmentMin) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(segment_ids)}, {3, INPUT_DESC(num_segments)}};
//将变量x、segment_ids、num_segments处理并存入对应InputDesc结构体的相应变量中，存入UnsortedSegmentMin对应空间内并用input_map指针保存
ATTR_MAP(UnsortedSegmentMin) = EMPTY_ATTR_MAP;//将空变量存入UnsortedSegmentMin对应空间并用attr_map_指针保存
OUTPUT_MAP(UnsortedSegmentMin) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入UnsortedSegmentMin对应空间内并用output_map_指针保存
REG_ADPT_DESC(UnsortedSegmentMin, prim::kPrimUnsortedSegmentMin->name(), ADPT_DESC(UnsortedSegmentMin))
//构造指向UnsortedSegmentMin的指针并储存，创建结构体RegAdptDescUnsortedSegmentMin

// ReverseV2
INPUT_MAP(ReverseV2D) = {{1, INPUT_DESC(x)}};//将变量x处理并存入对应InputDesc结构体的相应变量中，存入ReverseV2D对应空间内并用input_map指针保存
ATTR_MAP(ReverseV2D) = {{"axis", ATTR_DESC(axis, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())}};//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//                                                                                                            存入ReverseV2D对应空间并用attr_map_指针保存
//                                                                                                            其中AnyTraits<>的作用为将<>内类型进行构建
//                                                                                                            std:vector<>的作用为构建<>内类型的容量可变的数组
OUTPUT_MAP(ReverseV2D) = {{0, OUTPUT_DESC(y)}};//将变量y处理并存入对应OutputDesc结构体的相应变量中，存入ReverseV2D对应空间内并用output_map_指针保存
REG_ADPT_DESC(ReverseV2D, kNameReverseV2, ADPT_DESC(ReverseV2D))
//构造指向ReverseV2D的指针并储存，创建结构体RegAdptDescReverseV2D
}  // namespace mindspore::transform
