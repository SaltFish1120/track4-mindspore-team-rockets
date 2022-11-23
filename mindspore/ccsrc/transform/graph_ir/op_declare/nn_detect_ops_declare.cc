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

#include "transform/graph_ir/op_declare/nn_detect_ops_declare.h"
#include <vector>

namespace mindspore::transform {
// BoundingBoxEncode
INPUT_MAP(BoundingBoxEncode) = {
  {1, INPUT_DESC(anchor_box)},
  {2, INPUT_DESC(ground_truth_box)},
};//将BoundingBoxEncode类型与标准进行对比，后进行空间调整，并用指针input_map_为key存储相应内容
ATTR_MAP(BoundingBoxEncode) = {
  {"means", ATTR_DESC(means, AnyTraits<std::vector<float>>(), AnyTraits<float>())},
  {"stds", ATTR_DESC(stds, AnyTraits<std::vector<float>>(), AnyTraits<float>())},
};/*
将means处理并存入对应 ATTR_DESC结构体的相应变量中
将means内容转为字符串变量并存储至结构体的name变量中
引用Operator空间并将指针所指的类转为 ATTR_DESC结构体
//对相应变量处理并存入对应AttrDesc结构体的相应变量中
//存入MaxPool对应空间并用attr_map_指针保存
//其中AnyTraits<>的作用为将<>内类型进行构建
//std:vector<>的作用为构建<>内类型的容量可变的数组
*/
//将BoundingBoxEncode类型与标准进行对比，后进行空间调整，并用指针attr_map_为key存储相应内容
OUTPUT_MAP(BoundingBoxEncode) = {{0, OUTPUT_DESC(delats)}};//将delats处理并存入对应OUTPUT_DESC结构体的相应变量中
  //将delats内容转为字符串变量并存储至结构体的name变量中
  //引用Operator空间并将指针所指的类转为OUTPUT_DESC结构体
  //将BoundingBoxEncode类型与标准进行对比，后进行空间调整，并用指针output_map_为key存储相应内容
REG_ADPT_DESC(BoundingBoxEncode, kNameBoundingBoxEncode, ADPT_DESC(BoundingBoxEncode))//将BoudingBoxEncode处理并存入对应RED_ADPT_DESC结构体的相应变量中
  //将BoudingBoxEncode内容转为字符串变量并存储至结构体的name变量中
  //引用Operator空间并将指针所指的类转为RED_ADPT_DESC结构体


// BoundingBoxDecode
INPUT_MAP(BoundingBoxDecode) = {
  {1, INPUT_DESC(rois)},
  {2, INPUT_DESC(deltas)},
};
ATTR_MAP(BoundingBoxDecode) = {
  {"means", ATTR_DESC(means, AnyTraits<std::vector<float>>(), AnyTraits<float>())},
  {"stds", ATTR_DESC(stds, AnyTraits<std::vector<float>>(), AnyTraits<float>())},
  {"max_shape", ATTR_DESC(max_shape, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"wh_ratio_clip", ATTR_DESC(wh_ratio_clip, AnyTraits<float>())},
};
OUTPUT_MAP(BoundingBoxDecode) = {{0, OUTPUT_DESC(bboxes)}};
REG_ADPT_DESC(BoundingBoxDecode, kNameBoundingBoxDecode, ADPT_DESC(BoundingBoxDecode))

// Iou
INPUT_MAP(Iou) = {{1, INPUT_DESC(bboxes)}, {2, INPUT_DESC(gtboxes)}};
ATTR_MAP(Iou) = {{"mode", ATTR_DESC(mode, AnyTraits<std::string>())}};
OUTPUT_MAP(Iou) = {{0, OUTPUT_DESC(overlap)}};
REG_ADPT_DESC(Iou, kNameIOU, ADPT_DESC(Iou))

// CheckValid
INPUT_MAP(CheckValid) = {{1, INPUT_DESC(bbox_tensor)}, {2, INPUT_DESC(img_metas)}};
ATTR_MAP(CheckValid) = EMPTY_ATTR_MAP;
OUTPUT_MAP(CheckValid) = {{0, OUTPUT_DESC(valid_tensor)}};
REG_ADPT_DESC(CheckValid, kNameCheckValid, ADPT_DESC(CheckValid))

// Sort
INPUT_MAP(Sort) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Sort) = {{"axis", ATTR_DESC(axis, AnyTraits<int64_t>())},
                  {"descending", ATTR_DESC(descending, AnyTraits<bool>())}};
OUTPUT_MAP(Sort) = {{0, OUTPUT_DESC(y1)}, {1, OUTPUT_DESC(y2)}};
REG_ADPT_DESC(Sort, kNameSort, ADPT_DESC(Sort))

// ROIAlign
INPUT_MAP(ROIAlign) = {{1, INPUT_DESC(features)}, {2, INPUT_DESC(rois)}};
OUTPUT_MAP(ROIAlign) = {{0, OUTPUT_DESC(y)}};
ATTR_MAP(ROIAlign) = {{"pooled_height", ATTR_DESC(pooled_height, AnyTraits<int64_t>())},
                      {"pooled_width", ATTR_DESC(pooled_width, AnyTraits<int64_t>())},
                      {"spatial_scale", ATTR_DESC(spatial_scale, AnyTraits<float>())},
                      {"sample_num", ATTR_DESC(sample_num, AnyTraits<int64_t>())},
                      {"roi_end_mode", ATTR_DESC(roi_end_mode, AnyTraits<int64_t>())}};
REG_ADPT_DESC(ROIAlign, kNameROIAlign, ADPT_DESC(ROIAlign))
// ROIAlignGrad
INPUT_MAP(ROIAlignGrad) = {{1, INPUT_DESC(ydiff)}, {2, INPUT_DESC(rois)}};
OUTPUT_MAP(ROIAlignGrad) = {{0, OUTPUT_DESC(xdiff)}};
ATTR_MAP(ROIAlignGrad) = {
  {"xdiff_shape", ATTR_DESC(xdiff_shape, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())},
  {"pooled_height", ATTR_DESC(pooled_height, AnyTraits<int64_t>())},
  {"pooled_width", ATTR_DESC(pooled_width, AnyTraits<int64_t>())},
  {"spatial_scale", ATTR_DESC(spatial_scale, AnyTraits<float>())},
  {"sample_num", ATTR_DESC(sample_num, AnyTraits<int64_t>())}};
REG_ADPT_DESC(ROIAlignGrad, kNameROIAlignGrad, ADPT_DESC(ROIAlignGrad))
}  // namespace mindspore::transform
