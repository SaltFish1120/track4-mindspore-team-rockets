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

#include "minddata/mindrecord/include/shard_index.h"//按照路径寻找以下文件，导入到本文件

namespace mindspore {//创建名为mindspore的空间
namespace mindrecord {//创建名为mindrecord的空间
// table name for index索引的表名
const char TABLENAME[] = "index_table";
//调用Index类中Index函数给成员变量database_name_、table_name_赋值
Index::Index() : database_name_(""), table_name_(TABLENAME) {}
//创建Index空间下void型AddIndexField函数,调用fields_类中emplace_back函数
void Index::AddIndexField(const int64_t &schemaId, const std::string &field) {
  fields_.emplace_back(pair<int64_t, string>(schemaId, field));
}

// Get attribute list获取属性列表
std::vector<std::pair<uint64_t, std::string>> Index::GetFields() { return fields_; }
}  // namespace mindrecord
}  // namespace mindspore
