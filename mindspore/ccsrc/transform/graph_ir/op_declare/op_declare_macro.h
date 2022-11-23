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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_MACRO_H_//判断宏是否被定义，如果宏没有定义，则编译下面代码
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_MACRO_H_//定义预处理宏CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_MACRO_H_

//导入系统文件或自定义文件
#include <string>//导入标准库中的字符串类及相关操作
#include <memory>//提供make_shared指针构建函数模版等
#include "utils/hash_map.h"//按照路径寻找以下文件，导入到本文件，以下同理
#include "transform/graph_ir/op_adapter.h"
#include "transform/graph_ir/op_adapter_desc.h"
#include "include/transform/graph_ir/op_adapter_map.h"
#include "mindspore/core/base/core_ops.h"

namespace mindspore::transform {//创建名为transform的空间，其空间处于空间mindspore下
/*
  定义宏变量DECLARE_OP_ADAPTER
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
#define DECLARE_OP_ADAPTER(T)                                        \
  using T = ge::op::T;                                               \
  template <>                                                        \
  const mindspore::HashMap<int, InputDesc> OpAdapter<T>::input_map_; \
  template <>                                                        \
  const mindspore::HashMap<std::string, AttrDesc> OpAdapter<T>::attr_map_;
//定义宏变量DECLARE_OP_USE_OUTPUT
//操作同上，对象变为int+OutputDesc，用指针output_map_为key存储相应内容
#define DECLARE_OP_USE_OUTPUT(T) \
  template <>                    \
  const mindspore::HashMap<int, OutputDesc> OpAdapter<T>::output_map_;
//定义宏变量DECLARE_OP_USE_ENUM
//操作同上，对象变为int+string，用指针enum_map_为key存储相应内容
#define DECLARE_OP_USE_ENUM(T) \
  template <>                  \
  const mindspore::HashMap<std::string, int> OpAdapter<T>::enum_map_{};
//定义宏变量DECLARE_OP_USE_INPUT_ATTR
//操作同上，对象变为unsigned int+AttrDesc，用指针input_attr_map_为key存储相应内容
#define DECLARE_OP_USE_INPUT_ATTR(T) \
  template <>                        \
  const mindspore::HashMap<unsigned int, AttrDesc> OpAdapter<T>::input_attr_map_;
//定义宏变量DECLARE_OP_USE_DYN_INPUT
//操作同上，对象变为int+DynInputDesc，用指针dyn_input_map_为key存储相应内容
#define DECLARE_OP_USE_DYN_INPUT(T) \
  template <>                       \
  const mindspore::HashMap<int, DynInputDesc> OpAdapter<T>::dyn_input_map_;
//定义宏变量DECLARE_OP_USE_DYN_SUBGRAPH
//操作同上，对象变为int+DynSubGraphDesc，用指针dyn_subgraph_map_为key存储相应内容
#define DECLARE_OP_USE_DYN_SUBGRAPH(T) \
  template <>                          \
  const mindspore::HashMap<int, DynSubGraphDesc> OpAdapter<T>::dyn_subgraph_map_;
//定义宏变量DECLARE_OP_USE_DYN_OUTPUT
//操作同上，对象变为int+DynOutputDesc，用指针dyn_output_map_为key存储相应内容
#define DECLARE_OP_USE_DYN_OUTPUT(T) \
  template <>                        \
  const mindspore::HashMap<int, DynOutputDesc> OpAdapter<T>::dyn_output_map_;
//定义宏变量INPUT_MAP
//操作同上，对象变为int+InputDesc，用指针input_map_为key存储相应内容
#define INPUT_MAP(T) \
  template <>        \
  const mindspore::HashMap<int, InputDesc> OpAdapter<T>::input_map_
//定义宏变量EMPTY_INPUT_MAP
//操作同上，对象变为int+InputDesc，不使用指针进行储存
#define EMPTY_INPUT_MAP mindspore::HashMap<int, InputDesc>()
/*
  定义宏变量INPUT_DESC
  将name变量处理并存入对应InputDesc结构体的相应变量中
  将name变量内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为InputDesc结构体
  利用指针分别将原指向空间名称、handle变量指向空间+输出空间名称、引用的TensorDesc空间依次存入set_op、set_handle、update_input_desc变量中
*/
#define INPUT_DESC(name) \
  {                      \
#name, \
    [](const OperatorPtr op, const OperatorPtr input) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->set_input_##name(*input); \
    }, \
    [](const OperatorPtr op, const OutHandler& handle) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->set_input_##name(*(handle.op), handle.out); \
    }, \
    [](const OperatorPtr op, const GeTensorDesc desc) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->update_input_desc_##name(desc); \
    }                 \
  }
//定义宏变量DYN_INPUT_MAP
//操作同第一个函数，对象变为int+DynInputDesc，用指针dyn_input_map_为key存储相应内容
#define DYN_INPUT_MAP(T) \
  template <>            \
  const mindspore::HashMap<int, DynInputDesc> OpAdapter<T>::dyn_input_map_
/*
  定义宏变量DYN_INPUT_DESC
  将name变量处理并存入对应DynInputDesc结构体的相应变量中
  将name变量内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为DynInputDesc结构体
  利用指针分别将携带数字、携带数字+引用Operator空间的内容、数字+handle变量指向空间+输出空间名称依次存入create_dyn_input、set_op、set_handle变量中
*/
#define DYN_INPUT_DESC(name) \
  {                          \
#name, \
    [](const OperatorPtr op, unsigned int num) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->create_dynamic_input_##name(num); \
    }, \
    [](const OperatorPtr op, unsigned int index, const OperatorPtr input) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->set_dynamic_input_##name(index, *input); \
    }, \
    [](const OperatorPtr op, unsigned int index, const OutHandler& handle) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->set_dynamic_input_##name(index, *(handle.op), handle.out); \
    }                     \
  }
//定义宏变量DYN_SUBGRAPH_MAP
//操作同第一个函数，对象变为int+DynInputDesc，用指针dyn_subgraph_map_为key存储相应内容
#define DYN_SUBGRAPH_MAP(T) \
  template <>               \
  const mindspore::HashMap<int, DynSubGraphDesc> OpAdapter<T>::dyn_subgraph_map_
/*
  定义宏变量DYN_SUBGRAPH_DESC
  将name变量处理并存入对应DynSubgraphDesc结构体的相应变量中
  将name变量内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为DynSubgraphDesc结构体
  利用指针分别将携带数字、携带数字+Graph空间内容依次存入create_dyn_subgraph、set_subgraph变量中
*/
#define DYN_SUBGRAPH_DESC(name) \
  {                             \
#name, \
    [](const OperatorPtr op, unsigned int num) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->create_dynamic_subgraph_##name(num); \
    }, \
    [](const OperatorPtr op, unsigned int index, const DfGraphPtr graph) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->set_dynamic_subgraph_builder_##name(index, [graph](){return *graph;}); \
    }                        \
  }
//定义宏变量ATTR_MAP
//操作同第一个函数，对象变为string+AttrDesc，用指针attr_map_为key存储相应内容
#define ATTR_MAP(T) \
  template <>       \
  const mindspore::HashMap<std::string, AttrDesc> OpAdapter<T>::attr_map_
//定义宏变量EMPTY_ATTR_MAP
//操作同第一个函数，对象变为string+AttrDesc，不使用指针进行储存
#define EMPTY_ATTR_MAP mindspore::HashMap<std::string, AttrDesc>()
/*
  定义宏变量ATTR_DESC
  将name变量处理并存入对应AttrDesc结构体的相应变量中
  将name变量内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为AttrDesc结构体
  利用指针将空间存入set_attr变量中
*/
#define ATTR_DESC(name, ...) \
  {                          \
#name, \
    [](const OperatorPtr op, const ValuePtr& value) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->set_attr_##name(ConvertAny(value, __VA_ARGS__)); \
    }                     \
  }
//定义宏变量INPUT_ATTR_MAP 
//操作同第一个函数，对象变为unsigned int+AttrDesc，用指针input_attr_map_为key存储相应内容
#define INPUT_ATTR_MAP(T) \
  template <>             \
  const mindspore::HashMap<unsigned int, AttrDesc> OpAdapter<T>::input_attr_map_
//定义宏变量OUTPUT_MAP 
//操作同第一个函数，对象变为int+OutputDesc，用指针output_map_为key存储相应内容
#define OUTPUT_MAP(T) \
  template <>         \
  const mindspore::HashMap<int, OutputDesc> OpAdapter<T>::output_map_
/*
  定义宏变量OUTPUT_DESC
  将name变量处理并存入对应OutputDesc结构体的相应变量中
  将name变量内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为OutputDesc结构体
  利用指针将引用的TensorDesc空间存入update_out_desc变量中
*/
#define OUTPUT_DESC(name) \
  {                       \
#name, \
    [](const OperatorPtr op, const GeTensorDesc desc) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->update_output_desc_##name(desc); \
    }                  \
  }
//定义宏变量DYN_OUTPUT_MAP 
//操作同第一个函数，对象变为int+DynOutputDesc，用指针dyn_output_map_为key存储相应内容
#define DYN_OUTPUT_MAP(T) \
  template <>             \
  const mindspore::HashMap<int, DynOutputDesc> OpAdapter<T>::dyn_output_map_
/*
  定义宏变量DYN_OUTPUT_DESC
  将name变量处理并存入对应DynOutputDesc结构体的相应变量中
  将name变量内容转为字符串变量并存储至结构体的name变量中
  引用Operator空间并将指针所指的类转为DynOutputDesc结构体
  利用指针将携带数字存入create_dyn_output变量中
*/
#define DYN_OUTPUT_DESC(name) \
  {                           \
#name, \
    [](const OperatorPtr op, unsigned int num) { \
        auto p = std::static_pointer_cast<OpType>(op); \
        (void)p->create_dynamic_output_##name(num); \
    }                      \
  }
//定义宏变量ADPT_DESC_ONE
//分配并构造类型为OpAdapterDesc的对象，将类型为T、指向OpAdapter空间的指针传递给其构造函数，然后返回类型为shared_ptr<OpAdapterDesc>的对象，该对象拥有并存储指向它的指针
#define ADPT_DESC_ONE(T) std::make_shared<OpAdapterDesc>(std::make_shared<OpAdapter<T>>())
//定义宏变量ADPT_DESC_TWO
//分配并构造类型为OpAdapterDesc的对象，将类型为T、指向OpAdapter空间的指针+类型为I、指向OpAdapter空间的指针传递给其构造函数，然后返回类型为shared_ptr<OpAdapterDesc>的对象，该对象拥有并存储指向它的指针
#define ADPT_DESC_TWO(T, I) \
  std::make_shared<OpAdapterDesc>(std::make_shared<OpAdapter<T>>(), std::make_shared<OpAdapter<I>>())
//定义宏变量GET_DESC
//承担各形参变量的支架函数
#define GET_MACRO(_1, _2, DESC, ...) DESC
//定义宏变量ADPT_DESC
//将GET_MACRO宏变量中嵌套ADPT_DESC_ONE、ADPT_DESC_TWO等宏变量，构造对应空间指针
#define ADPT_DESC(...) GET_MACRO(__VA_ARGS__, ADPT_DESC_TWO, ADPT_DESC_ONE, ...)(__VA_ARGS__)
/*
  定义宏变量REG_ADPT_DESC
  创建结构体RegAdptDesc+name
  创建公共型函数RegAdptDesc+name：将访问OpAdapterDesc空间的内容与string进行对比，后进行空间调整，并用指针adpt_desc指针存储相应内容
  创建私密性整形变量：ph_{0}
  将其结构体自定义命名为g_reg_adpt_desc_+name
*/
#define REG_ADPT_DESC(name, name_str, adpt_desc)                       \
  static struct RegAdptDesc##name {                                    \
   public:                                                             \
    RegAdptDesc##name() { OpAdapterMap::get()[name_str] = adpt_desc; } \
                                                                       \
   private:                                                            \
    int ph_{0};                                                        \
  } g_reg_adpt_desc_##name;
}  // namespace mindspore::transform
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_MACRO_H_
