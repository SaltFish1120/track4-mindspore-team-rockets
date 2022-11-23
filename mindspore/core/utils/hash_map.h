/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_UTILS_HASH_MAP_H_
#define MINDSPORE_CORE_UTILS_HASH_MAP_H_

#include <functional>
#if (ENABLE_FAST_HASH_TABLE) && __has_include("robin_hood/robin_hood.h")
#include "robin_hood/robin_hood.h"

namespace mindspore {
//Hash作用：将算子类型强制转化为K，KeyEqual作用：判断算子类型是否成功转化为K
//判断T类型是否为void，若是则返回类型Key，若不是则根据IsFlat的真假返回Key或Key const，并与T类型交换最后返回其值。（获得最终的类型）
//判断Key与T的容量是否小于标准+判断Key与T是否为可移动构造类型+判断Key与T是否拥有移动赋值运算符，获得IsFlat
//Table作用：判断完成后，若结果为真，将范围在4~ 16384的内容从空间分配器中删去
//                     若结果为假，建立新的分配器并删除原分配的空间（链表），建立新的内存块并加入到新的分配器中。
//                     统计可用元素数量，通过已分配空间计算需要分配的内存大小
//                     为新数据创建链表并加入至分配的空间中，元素字节强制对齐。
//并在建成的堆中分配具体空间
template <typename K, typename V, typename Hash = robin_hood::hash<K>, typename KeyEqual = std::equal_to<K>>
using HashMap = robin_hood::unordered_map<K, V, Hash, KeyEqual>;

#else
#include <unordered_map>

namespace mindspore {
template <typename K, typename V, typename Hash = std::hash<K>, typename KeyEqual = std::equal_to<K>>
using HashMap = std::unordered_map<K, V, Hash, KeyEqual>;

#endif
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_HASH_MAP_H_
