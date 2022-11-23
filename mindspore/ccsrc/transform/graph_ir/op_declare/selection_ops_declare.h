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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_SELECTION_OPS_DECLARE_H_//判断宏是否被定义，如果宏没有定义，则编译下面代码
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_SELECTION_OPS_DECLARE_H_//定义预处理宏_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_SELECTION_DECLARE_H_

#include <string>//导入标准库中的字符串类及相关操作
#include "utils/hash_map.h"//按照路径寻找以下文件，导入到本文件,以下同理
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "ops/selection_ops.h"

namespace mindspore::transform {//创建名为transform的空间，其空间处于空间mindspore下
//对实现SliceD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(SliceD)
DECLARE_OP_USE_INPUT_ATTR(SliceD)
DECLARE_OP_USE_OUTPUT(SliceD)
//对实现ScatterNdD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(ScatterNdD)
DECLARE_OP_USE_INPUT_ATTR(ScatterNdD)
DECLARE_OP_USE_OUTPUT(ScatterNdD)
//对实现ScatterNonAliasingAdd的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(ScatterNonAliasingAdd)
DECLARE_OP_USE_OUTPUT(ScatterNonAliasingAdd)
//对实现GatherNd的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(GatherNd)
DECLARE_OP_USE_OUTPUT(GatherNd)
//对实现GatherD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(GatherD)
DECLARE_OP_USE_OUTPUT(GatherD)
//对实现TopK的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(TopK)
DECLARE_OP_USE_OUTPUT(TopK)
//对实现InTopKD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(InTopKD)
DECLARE_OP_USE_OUTPUT(InTopKD)
//对实现Select的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(Select)
DECLARE_OP_USE_OUTPUT(Select)
//对实现StridedSliceGrad的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(StridedSliceGrad)
DECLARE_OP_USE_OUTPUT(StridedSliceGrad)
//对实现StridedSlice的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(StridedSlice)
DECLARE_OP_USE_OUTPUT(StridedSlice)
//对实现StridedSliceV2的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(StridedSliceV2)
DECLARE_OP_USE_OUTPUT(StridedSliceV2)
//对实现UnsortedSegmentSumD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(UnsortedSegmentSumD)
DECLARE_OP_USE_INPUT_ATTR(UnsortedSegmentSumD)
DECLARE_OP_USE_OUTPUT(UnsortedSegmentSumD)
//对实现UnsortedSegmentProdD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(UnsortedSegmentProdD)
DECLARE_OP_USE_INPUT_ATTR(UnsortedSegmentProdD)
DECLARE_OP_USE_OUTPUT(UnsortedSegmentProdD)
//对实现UnsortedSegmentMaxD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(UnsortedSegmentMaxD)
DECLARE_OP_USE_INPUT_ATTR(UnsortedSegmentMaxD)
DECLARE_OP_USE_OUTPUT(UnsortedSegmentMaxD)
//对实现UnsortedSegmentMin的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(UnsortedSegmentMin)
DECLARE_OP_USE_OUTPUT(UnsortedSegmentMin)
//对实现CumprodD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(CumprodD)
DECLARE_OP_USE_INPUT_ATTR(CumprodD)
DECLARE_OP_USE_OUTPUT(CumprodD)
//对实现TileD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(TileD)
DECLARE_OP_USE_INPUT_ATTR(TileD)
DECLARE_OP_USE_OUTPUT(TileD)
//对实现OneHot的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(OneHot)
DECLARE_OP_USE_OUTPUT(OneHot)
//对实现GatherV2D的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(GatherV2D)
DECLARE_OP_USE_INPUT_ATTR(GatherV2D)
DECLARE_OP_USE_OUTPUT(GatherV2D)
//对实现RangeD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(RangeD)
DECLARE_OP_USE_OUTPUT(RangeD)
//对实现InplaceAddD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(InplaceAddD)
DECLARE_OP_USE_OUTPUT(InplaceAddD)
//对实现InplaceSubD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(InplaceSubD)
DECLARE_OP_USE_OUTPUT(InplaceSubD)
//对实现InplaceUpdateD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(InplaceUpdateD)
DECLARE_OP_USE_OUTPUT(InplaceUpdateD)
//对实现CumsumD的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(CumsumD)
DECLARE_OP_USE_INPUT_ATTR(CumsumD)
DECLARE_OP_USE_OUTPUT(CumsumD)
//对实现GatherV2的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(GatherV2)
DECLARE_OP_USE_OUTPUT(GatherV2)
//对实现ReverseV2D的对象转换后转移出相应存储空间
DECLARE_OP_ADAPTER(ReverseV2D)
DECLARE_OP_USE_OUTPUT(ReverseV2D)
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_SELECTION_OPS_DECLARE_H_
