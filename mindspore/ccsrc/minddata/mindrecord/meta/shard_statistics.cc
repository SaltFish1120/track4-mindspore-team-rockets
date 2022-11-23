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

#include "minddata/mindrecord/include/shard_statistics.h"//按照路径寻找以下文件，导入到本文件,以下同理
#include "pybind11/pybind11.h"

using mindspore::LogStream;//声明mindspore空间下的LogStream
using mindspore::ExceptionType::NoExceptionType;//声明mindspore空间下ExceptionType类中的NoExceptionType
using mindspore::MsLogLevel::ERROR;//声明mindspore空间下MsLogLevel类中的ERROR

namespace mindspore {//创建名为mindspore的空间
namespace mindrecord {//创建名为mindrecord的空间
//建立存储空间的函数
//引用Schema空间中的Build函数，返回空间指针
std::shared_ptr<Statistics> Statistics::Build(std::string desc, const json &statistics) {
  // validate check验证检查
  if (!Validate(statistics)) {
    return nullptr;
  }
  Statistics object_statistics;
  object_statistics.desc_ = std::move(desc);
  object_statistics.statistics_ = statistics;
  object_statistics.statistics_id_ = -1;
  return std::make_shared<Statistics>(object_statistics);
}
//引用Statistics空间中的GetDesc函数，返回desc_参数
std::string Statistics::GetDesc() const { return desc_; }
//引用Statistics空间中GetStatistics函数，返回str_schema数组
//str_statistics数组中储存这statistics的基本数值
json Statistics::GetStatistics() const {
  json str_statistics;
  str_statistics["desc"] = desc_;
  str_statistics["statistics"] = statistics_;
  return str_statistics;
}
//引用Statistics空间中SetStatisticsID函数
//使statistics_id_储存相应的id数值
void Statistics::SetStatisticsID(int64_t id) { statistics_id_ = id; }
//引用Statistics空间中GetStatisticsID函数
//获得statistics_id_的值
int64_t Statistics::GetStatisticsID() const { return statistics_id_; }
//引用Statistics空间中的Validate函数
bool Statistics::Validate(const json &statistics) {
  //判断数据是否符合标准，若不符则输出错误信息，并返回false
  if (statistics.size() != kInt1) {
    MS_LOG(ERROR) << "Invalid data, 'statistics' is empty.";
    return false;
  }
  //判断数据是否是statistics空间的结尾，若是则输出错误信息，并返回false
  if (statistics.find("level") == statistics.end()) {
    MS_LOG(ERROR) << "Invalid data, 'level' object can not found in statistic";
    return false;
  }
  //若以上都不符合则返回LevelRecursive函数
  return LevelRecursive(statistics["level"]);
}
//引用Statistics空间中的LevelRecursive函数
bool Statistics::LevelRecursive(json level) {
  bool ini = true;
  //进入循环，遍历level数组
  for (json::iterator it = level.begin(); it != level.end(); ++it) {
    json a = it.value();
    //判断该空间大小是否符合标准2，若符合则判断该空间的key和count是否与无数据空间一致，若是则返回错误信息并返回false
    //若空间大小不符合标准2，则判断空间大小是否符合标准3，若符合则判断该空间key、count、level是否与无数据空间一致，若是则返回错误信息并返回false，若不符合则给ini变量赋值
    //若均不符合，则返回错误信息，返回false
    //最后返回ini的值
    if (a.size() == kInt2) {
      if ((a.find("key") == a.end()) || (a.find("count") == a.end())) {
        MS_LOG(ERROR) << "Invalid data, the node field is 2, but 'key'/'count' object does not existed";
        return false;
      }
    } else if (a.size() == kInt3) {
      if ((a.find("key") == a.end()) || (a.find("count") == a.end()) || a.find("level") == a.end()) {
        MS_LOG(ERROR) << "Invalid data, the node field is 3, but 'key'/'count'/'level' object does not existed";
        return false;
      } else {
        ini = LevelRecursive(a.at("level"));
      }
    } else {
      MS_LOG(ERROR) << "Invalid data, the node field is not equal to 2 or 3";
      return false;
    }
  }
  return ini;
}
//重载mindrecord空间中的Statistics中的b
//加入判断：判断此空间下GetStatistics函数和GetStatistics函数的返回值是否与b的相应函数的返回值相同
//任意等式不成立，则返回false，反之返回true
bool Statistics::operator==(const Statistics &b) const {
  if (this->GetStatistics() != b.GetStatistics()) {
    return false;
  }
  return true;
}
}  // namespace mindrecord
}  // namespace mindspore
