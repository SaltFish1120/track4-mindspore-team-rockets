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

#include "minddata/mindrecord/include/shard_schema.h"//按照路径寻找以下文件，导入到本文件,以下同理
#include "utils/ms_utils.h"

using mindspore::LogStream;//声明mindspore空间下的LogStream
using mindspore::ExceptionType::NoExceptionType;//声明mindspore空间下ExceptionType类中的NoExceptionType
using mindspore::MsLogLevel::ERROR;//声明mindspore空间下MsLogLevel类中的ERROR

namespace mindspore {//创建名为mindspore的空间
namespace mindrecord {//创建名为mindrecord的空间
//建立存储空间的函数
//引用Schema空间中的Build函数，返回空间指针
std::shared_ptr<Schema> Schema::Build(std::string desc, const json &schema) {
  // validate check验证检查
  if (!Validate(schema)) {
    return nullptr;
  }

  std::vector<std::string> blob_fields = PopulateBlobFields(schema);
  Schema object_schema;
  object_schema.desc_ = std::move(desc);
  object_schema.blob_fields_ = std::move(blob_fields);
  object_schema.schema_ = schema;
  object_schema.schema_id_ = -1;
  return std::make_shared<Schema>(object_schema);
}
//引用Schema空间中的GetDesc函数，返回desc_参数
std::string Schema::GetDesc() const { return desc_; }
//引用Schema空间中GetSchema函数，返回str_schema数组
//str_schema数组中储存这Schema的基本数值
json Schema::GetSchema() const {
  json str_schema;
  str_schema["desc"] = desc_;
  str_schema["schema"] = schema_;
  str_schema["blob_fields"] = blob_fields_;
  return str_schema;
}
//引用Schema空间中SetSchemaID函数
//使schema_id_储存相应的id数值
void Schema::SetSchemaID(int64_t id) { schema_id_ = id; }
//引用Schema空间中GetSchemaID函数
//获得schema_id_的值
int64_t Schema::GetSchemaID() const { return schema_id_; }
//引用Schema空间中GetGetBlobFields函数
//获得blob_fields_的值
std::vector<std::string> Schema::GetBlobFields() const { return blob_fields_; }
//引用Schema空间中的PopulateBlobFields函数
//依次对比schema数组中各项的size、shape、type是否符合要求，若符合则将该项的key存储进blob_fields
//最终返回blob_fields
std::vector<std::string> Schema::PopulateBlobFields(json schema) {
  std::vector<std::string> blob_fields;
  for (json::iterator it = schema.begin(); it != schema.end(); ++it) {
    json it_value = it.value();
    if ((it_value.size() == kInt2 && it_value.find("shape") != it_value.end()) || it_value["type"] == "bytes") {
      blob_fields.emplace_back(it.key());
    }
  }
  return blob_fields;
}
//引用Schema空间中的ValidateNumberShape函数
bool Schema::ValidateNumberShape(const json &it_value) {
  //判断传入的数据是否是数组的结尾，若是则输出错误信息，并返回false
  if (it_value.find("shape") == it_value.end()) {
    MS_LOG(ERROR) << "Invalid schema, 'shape' object can not found in " << it_value.dump()
                  << ". Please check the input schema.";
    return false;
  }
  //给shape变量赋值，值为输入数据的类型
  //判断其值是否在标准范围内，若不是则输出错误信息，并返回false
  auto shape = it_value["shape"];
  if (!shape.is_array()) {
    MS_LOG(ERROR) << "Invalid schema, the value of 'shape' should be list format but got: " << it_value["shape"]
                  << ". Please check the input schema.";
    return false;
  }
  //给num_negtive_one变量赋值为0
  //进入循环并不断给i进行赋值，值为不同的shape
  //若出现i等于0或i小于-1的情况，输出错误信息，并返回false
  //若出现i等于-1的情况，则num_negtive_one+1进行计数
  int num_negtive_one = 0;
  for (const auto &i : shape) {
    if (i == 0 || i < -1) {
      MS_LOG(ERROR) << "Invalid schema, the element of 'shape' value should be -1 or greater than 0 but got: " << i
                    << ". Please check the input schema.";
      return false;
    }
    if (i == -1) {
      num_negtive_one++;
    }
  }
  //判断num_negtive_one是否大于1，若是则输出错误信息，并返回false
  if (num_negtive_one > 1) {
    MS_LOG(ERROR) << "Invalid schema, only 1 variable dimension(-1) allowed in 'shape' value but got: "
                  << it_value["shape"] << ". Please check the input schema.";
    return false;
  }

  return true;
}
////引用Schema空间中的Validate函数
bool Schema::Validate(json schema) {
  //判断schema空间是否为空，若是则输出错误信息，并返回false
  if (schema.empty()) {
    MS_LOG(ERROR) << "Invalid schema, schema is empty. Please check the input schema.";
    return false;
  }
  //进入循环，依次判断schema空间中所有数据是否符合标准，若不是则输出错误信息，并返回false
  for (json::iterator it = schema.begin(); it != schema.end(); ++it) {
    // make sure schema key name must be composed of '0-9' or 'a-z' or 'A-Z' or '_'确保架构密钥名称必须由“0-9”、“a-z”、“a-z”或“_”组成
    if (!ValidateFieldName(it.key())) {
      MS_LOG(ERROR) << "Invalid schema, field name: " << it.key()
                    << "is not composed of '0-9' or 'a-z' or 'A-Z' or '_'. Please rename the field name in schema.";
      return false;
    }
    //确保数据的type存在
    json it_value = it.value();
    if (it_value.find("type") == it_value.end()) {
      MS_LOG(ERROR) << "Invalid schema, 'type' object can not found in field " << it_value.dump()
                    << ". Please add the 'type' object for field in schema.";
      return false;
    }
    //确保数据的type合法
    if (kFieldTypeSet.find(it_value["type"]) == kFieldTypeSet.end()) {
      MS_LOG(ERROR) << "Invalid schema, the value of 'type': " << it_value["type"]
                    << " is not supported.\nPlease modify the value of 'type' to 'int32', 'int64', 'float32', "
                       "'float64', 'string', 'bytes' in schema.";
      return false;
    }
    //确保数据的合法数据运行后报错
    if (it_value.size() == kInt1) {
      continue;
    }
    //确保schema空间中存储空间为合法空间
    if (it_value["type"] == "bytes" || it_value["type"] == "string") {
      MS_LOG(ERROR)
        << "Invalid schema, no other field can be added when the value of 'type' is 'string' or 'types' but got: "
        << it_value.dump() << ". Please remove other fields in schema.";
      return false;
    }
    //确保schema空间中存储空间的type和shape属性完整
    if (it_value.size() != kInt2) {
      MS_LOG(ERROR) << "Invalid schema, the fields should be 'type' or 'type' and 'shape' but got: " << it_value.dump()
                    << ". Please check the schema.";
      return false;
    }
    //确保数据的shape属性符合条件
    if (!ValidateNumberShape(it_value)) {
      return false;
    }
  }

  return true;
}
//重载mindrecord空间中的Schema中的b
//加入判断：判断此空间下GetDesc函数和GetSchema函数的返回值是否与b的相应函数的返回值相同
//任意等式不成立，则返回false，反之返回true
bool Schema::operator==(const mindrecord::Schema &b) const {
  if (this->GetDesc() != b.GetDesc() || this->GetSchema() != b.GetSchema()) {
    return false;
  }
  return true;
}
}  // namespace mindrecord
}  // namespace mindspore
