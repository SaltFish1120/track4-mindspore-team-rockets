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

#include "minddata/mindrecord/include/shard_category.h"//按照路径寻找以下文件，导入到本文件

namespace mindspore {//创建名为mindspore的空间
namespace mindrecord {//创建名为mindrecord的空间
//引用ShardCategory空间中ShardCategory函数,输入三种参数调用指定函数
//对参数categories、category_field_、num_elements_、num_categories_、replacement_进行进行参数初始化
ShardCategory::ShardCategory(const std::vector<std::pair<std::string, std::string>> &categories, int64_t num_elements,
                             bool replacement)
    : categories_(categories),
      category_field_(""),
      num_elements_(num_elements),
      num_categories_(0),
      replacement_(replacement) {}
//引用ShardCategory空间中ShardCategory函数,输入四种参数调用指定函数
//对参数categories、category_field_、num_elements_、num_categories_、replacement_进行进行参数初始化
ShardCategory::ShardCategory(const std::string &category_field, int64_t num_elements, int64_t num_categories,
                             bool replacement)
    : categories_({}),
      category_field_(category_field),
      num_elements_(num_elements),
      num_categories_(num_categories),
      replacement_(replacement) {}

Status ShardCategory::Execute(ShardTaskList &tasks) { return Status::OK(); }//在ShardCategory空间中创建Status型Execute函数,返回值为Status空间中的OK函数的返回值
//在ShardCategory空间中创建int64_t型GetNumSamples函数,返回值为0或-1
//判断dataset_size的值
//若dataset_size的值为0,则返回dataset_size本身
//若dataset_size的值大于0,则继续判断num_classes、num_categories_、num_elements_的值是否大于0。若均大于0,则修改num_classes的赋值,赋值为num_categories_和num_classes的最小值
//继续判断num_classes是否为0,若为0则返回0。不为0则判断num_elements_是否大于int64_t类型最大值与num_classes的商,若是则返回-1。若均不符合则返回num_classes和num_elements_的乘积
//若均不符合,则返回0
int64_t ShardCategory::GetNumSamples(int64_t dataset_size, int64_t num_classes) {
  if (dataset_size == 0) return dataset_size;
  if (dataset_size > 0 && num_classes > 0 && num_categories_ > 0 && num_elements_ > 0) {
    num_classes = std::min(num_categories_, num_classes);
    if (num_classes == 0) {
      return 0;
    }
    if (num_elements_ > std::numeric_limits<int64_t>::max() / num_classes) {
      return -1;
    }
    return num_classes * num_elements_;
  }
  return 0;
}
}  // namespace mindrecord
}  // namespace mindspore
