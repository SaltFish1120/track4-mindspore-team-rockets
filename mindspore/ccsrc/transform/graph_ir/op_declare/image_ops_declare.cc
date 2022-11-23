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

#include "transform/graph_ir/op_declare/image_ops_declare.h"
#include <vector>

namespace mindspore::transform {
// ResizeNearestNeighborV2D
INPUT_MAP(ResizeNearestNeighborV2D) = {{1, INPUT_DESC(x)}};/*
    将x处理并存入对应INPUT_DESC结构体的相应变量中
  将x内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为INPUT_DESC结构体
将ResizeNearestNeighborV2D的类型与标准进行对比，后进行空间调整，并用指针input_map_为key存储相应内容
*/
ATTR_MAP(ResizeNearestNeighborV2D) = {
  {"size", ATTR_DESC(size, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())}};/*
  将size处理并存入对应ATTR_DESC结构体的相应变量中
  将size内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为ATTR_DESC结构体
//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//存入MaxPool对应空间并用attr_map_指针保存
//其中AnyTraits<>的作用为将<>内类型进行构建
//std:vector<>的作用为构建<>内类型的容量可变的数组
将ResizeNearestNeighborV2D的类型与标准进行对比，后进行空间调整，并用指针attr_map_为key存储相应内容
*/
OUTPUT_MAP(ResizeNearestNeighborV2D) = {{0, OUTPUT_DESC(y)}};/*
    将y处理并存入对应OUTPUT_DESC结构体的相应变量中
  将x内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为OUTPUT_DESC结构体
将ResizeNearestNeighborV2D的类型与标准进行对比，后进行空间调整，并用指针output_map_为key存储相应内容
*/
REG_ADPT_DESC(ResizeNearestNeighborV2D, kNameResizeNearestNeighborD, ADPT_DESC(ResizeNearestNeighborV2D))/*
  将ResizeNearestNeighborV2D处理并存入对应ADPT_DESC结构体的相应变量中
  将ResizeNearestNeighborV2D内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为ADPT_DESC结构体
  
  再将ResizeNearestNeighborV2D处理并存入对应REG_ADPT_DESC结构体的相应变量中
  将ResizeNearestNeighborV2D内容转为字符串变量并存储至REG_ADPT_DESC的结构体的name变量中
  引用Operator空间并将指针所指的类转为REG_ADPT_DESC结构体
*/          

// ResizeNearestNeighborV2
INPUT_MAP(ResizeNearestNeighborV2) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(size)}};
ATTR_MAP(ResizeNearestNeighborV2) = {{"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())},
                                     {"half_pixel_centers", ATTR_DESC(half_pixel_centers, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeNearestNeighborV2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ResizeNearestNeighborV2, kNameResizeNearestNeighborV2, ADPT_DESC(ResizeNearestNeighborV2))

// ResizeNearestNeighborV2Grad
INPUT_MAP(ResizeNearestNeighborV2Grad) = {{1, INPUT_DESC(grads)}, {2, INPUT_DESC(size)}};
ATTR_MAP(ResizeNearestNeighborV2Grad) = {{"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeNearestNeighborV2Grad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ResizeNearestNeighborV2Grad, kNameResizeNearestNeighborGrad, ADPT_DESC(ResizeNearestNeighborV2Grad))

// ResizeBilinearV2Grad
INPUT_MAP(ResizeBilinearV2Grad) = {{1, INPUT_DESC(grads)}, {2, INPUT_DESC(original_image)}};
ATTR_MAP(ResizeBilinearV2Grad) = {{"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeBilinearV2Grad) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ResizeBilinearV2Grad, kNameResizeBilinearGrad, ADPT_DESC(ResizeBilinearV2Grad))

// ResizeBilinearV2D
INPUT_MAP(ResizeBilinearV2D) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ResizeBilinearV2D) = {
  {"size", ATTR_DESC(size, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"align_corners", ATTR_DESC(align_corners, AnyTraits<bool>())}};
OUTPUT_MAP(ResizeBilinearV2D) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ResizeBilinearV2D, kNameResizeBilinear, ADPT_DESC(ResizeBilinearV2D))

// CropAndResize
INPUT_MAP(CropAndResize) = {
  {1, INPUT_DESC(x)}, {2, INPUT_DESC(boxes)}, {3, INPUT_DESC(box_index)}, {4, INPUT_DESC(crop_size)}};
ATTR_MAP(CropAndResize) = {{"extrapolation_value", ATTR_DESC(extrapolation_value, AnyTraits<float>())},
                           {"method", ATTR_DESC(method, AnyTraits<std::string>())}};
OUTPUT_MAP(CropAndResize) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(CropAndResize, kNameCropAndResize, ADPT_DESC(CropAndResize))
}  // namespace mindspore::transform
