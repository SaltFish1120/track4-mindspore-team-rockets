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

#include "minddata/mindrecord/include/shard_page.h"//按照路径寻找以下文件，导入到本文件,以下同理
#include "pybind11/pybind11.h"

namespace mindspore {//创建名为mindspore的空间
namespace mindrecord {//创建名为mindrecord的空间
//调用json Page类下GetPage函数,进行变量初始化
//判断函数row_group_ids_.size()的返回值是否为0，若是则将row_groups数组中"id"和"offset"的值设为0，若不是则分别设为rg.first和rg.second
//返回str_page
json Page::GetPage() const {
  json str_page;
  str_page["page_id"] = page_id_;
  str_page["shard_id"] = shard_id_;
  str_page["page_type"] = page_type_;
  str_page["page_type_id"] = page_type_id_;
  str_page["start_row_id"] = start_row_id_;
  str_page["end_row_id"] = end_row_id_;
  if (row_group_ids_.size() == 0) {
    json row_groups = json({});
    row_groups["id"] = 0;
    row_groups["offset"] = 0;
    str_page["row_group_ids"].push_back(row_groups);
  } else {
    for (const auto &rg : row_group_ids_) {
      json row_groups = json({});
      row_groups["id"] = rg.first;
      row_groups["offset"] = rg.second;
      str_page["row_group_ids"].push_back(row_groups);
    }
  }
  str_page["page_size"] = page_size_;
  return str_page;
}
//调用Page类下void型DeleteLastGroupId函数
//判断row_group_ids_.empty()函数的返回值是否为0，若是则修改page_size_的值为row_group_ids_.back().second，调用row_group_ids_.pop_back函数
void Page::DeleteLastGroupId() {
  if (!row_group_ids_.empty()) {
    page_size_ = row_group_ids_.back().second;
    row_group_ids_.pop_back();
  }
}
}  // namespace mindrecord
}  // namespace mindspore
