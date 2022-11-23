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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_CONTROL_FLOW_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_CONTROL_FLOW_OPS_DECLARE_H_

#include <string>
#include "utils/hash_map.h"
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "ops/control_flow_ops.h"

namespace mindspore::transform {
DECLARE_OP_ADAPTER(Merge)/*
  函数功能为将收录内容的类型与标准进行对比，后进行空间调整，并用指针input_map_为key存储相应内容
  具体实现过程：
    判断int与类InputDesc的容量是否小于标准+判断int与类InputDesc是否为可移动构造类型+判断int与类InputDesc是否拥有移动赋值运算符，获得IsFlat
    判断T类型是否为void，若是则返回类型Key，若不是则根据IsFlat的真假返回Key或Key const，并与T类型交换最后返回其值。
    判断完成后，若结果为真，将范围在4~ 16384的内容从空间分配器中删去
               若结果为假，建立新的分配器并删除原分配的空间（链表），建立新的内存块并加入到新的分配器中。
                          统计可用元素数量，通过已分配空间计算需要分配的内存大小
                          为新数据创建链表并加入至分配的空间中，元素字节强制对齐。
                          并在建成的堆中分配具体空间
    使用op_adapter_base.h下ge空间中op类下变量T，利用op_adapter.h中OpAdapter中分配的初始指针作为key储存元素
    string与AttrDesc进行相同操作，用指针attr_map_为key存储相应内容
*/
//此处变量为Merge，最终用input_map_为key储存相应内容
DECLARE_OP_USE_DYN_INPUT(Merge)//函数功能为将Merge的类型与标准进行对比，后进行空间调整，并用指针input_map_为key存储相应内容
DECLARE_OP_USE_OUTPUT(Merge)//函数功能为将Merge的类型与标准进行对比，后进行空间调整，并用指针output_map_为key存储相应内容

DECLARE_OP_ADAPTER(Switch)
DECLARE_OP_USE_OUTPUT(Switch)
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_CONTROL_FLOW_OPS_DECLARE_H_
