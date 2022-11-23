/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "minddata/mindrecord/include/shard_column.h"//按照路径寻找以下文件，导入到本文件,以下同理
#include "utils/ms_utils.h"
#include "minddata/mindrecord/include/common/shard_utils.h"
#include "minddata/mindrecord/include/shard_error.h"

namespace mindspore {//创建名为mindspore的空间
namespace mindrecord {//创建名为mindrecord的空间
//引用ShardColumn空间中ShardColumn函数,输入两种参数调用指定函数
//建立schema列表的表头
ShardColumn::ShardColumn(const std::shared_ptr<ShardHeader> &shard_header, bool compress_integer) {
  auto first_schema = shard_header->GetSchemas()[0];
  json schema_json = first_schema->GetSchema();
  Init(schema_json, compress_integer);
}
//引用ShardColumn空间中Execute函数
//引用Init函数进行操作
ShardColumn::ShardColumn(const json &schema_json, bool compress_integer) { Init(schema_json, compress_integer); }
//引用ShardColumn空间中Init函数
void ShardColumn::Init(const json &schema_json, bool compress_integer) {
  auto schema = schema_json["schema"];
  auto blob_fields = schema_json["blob_fields"];
  //进入循环，遍历schema列表，记录key变量
  bool has_integer_array = false;
  for (json::iterator it = schema.begin(); it != schema.end(); ++it) {
    const std::string &column_name = it.key();
    column_name_.push_back(column_name);

    json it_value = it.value();
    //判断shape是否为it_value列表的最末端，若不是则复制it_value列表加入column_shape_中，继续判断str_type是否是int32或int64类型，若是则记has_integer_array为真
    //若不是则将vec转换为int64并记入column_shape_列表中
    std::string str_type = it_value["type"];
    column_data_type_.push_back(ColumnDataTypeMap.at(str_type));
    if (it_value.find("shape") != it_value.end()) {
      std::vector<int64_t> vec(it_value["shape"].size());
      std::copy(it_value["shape"].begin(), it_value["shape"].end(), vec.begin());
      column_shape_.push_back(vec);
      if (str_type == "int32" || str_type == "int64") {
        has_integer_array = true;
      }
    } else {
      std::vector<int64_t> vec = {};
      column_shape_.push_back(vec);
    }
  }
  //创建column_name_id_列表
  for (uint64_t i = 0; i < column_name_.size(); i++) {
    column_name_id_[column_name_[i]] = i;
  }
  //创建blob_column_列表
  for (const auto &field : blob_fields) {
    blob_column_.push_back(field);
  }
  //创建blob_column_id_列表
  for (uint64_t i = 0; i < blob_column_.size(); i++) {
    blob_column_id_[blob_column_[i]] = i;
  }

  has_compress_blob_ = (compress_integer && has_integer_array);
  num_blob_column_ = blob_column_.size();
}
//引用ShardColumn空间中GetColumnTypeByName函数
Status ShardColumn::GetColumnTypeByName(const std::string &column_name, ColumnDataType *column_data_type,
                                        uint64_t *column_data_type_size, std::vector<int64_t> *column_shape,
                                        ColumnCategory *column_category) {
  RETURN_UNEXPECTED_IF_NULL(column_data_type);
  RETURN_UNEXPECTED_IF_NULL(column_data_type_size);
  RETURN_UNEXPECTED_IF_NULL(column_shape);
  RETURN_UNEXPECTED_IF_NULL(column_category);
  // Skip if column not found如果找不到列，则跳过
  *column_category = CheckColumnName(column_name);
  CHECK_FAIL_RETURN_UNEXPECTED(*column_category != ColumnNotFound,
                               "[Internal ERROR] the type of column: " + column_name + " can not found.");

  // Get data type and size获取数据类型和大小
  auto column_id = column_name_id_[column_name];
  *column_data_type = column_data_type_[column_id];
  *column_data_type_size = ColumnDataTypeSize[*column_data_type];
  *column_shape = column_shape_[column_id];
  return Status::OK();
}
//引用ShardColumn空间中GetColumnValueByName函数
Status ShardColumn::GetColumnValueByName(const std::string &column_name, const std::vector<uint8_t> &columns_blob,
                                         const json &columns_json, const unsigned char **data,
                                         std::unique_ptr<unsigned char[]> *data_ptr, uint64_t *const n_bytes,
                                         ColumnDataType *column_data_type, uint64_t *column_data_type_size,
                                         std::vector<int64_t> *column_shape) {
  RETURN_UNEXPECTED_IF_NULL(column_data_type);
  RETURN_UNEXPECTED_IF_NULL(column_data_type_size);
  RETURN_UNEXPECTED_IF_NULL(column_shape);
  // Skip if column not found如果找不到列，则跳过
  auto column_category = CheckColumnName(column_name);
  CHECK_FAIL_RETURN_UNEXPECTED(column_category != ColumnNotFound,
                               "[Internal ERROR] the type of column: " + column_name + " can not found.");
  // Get data type and size获取数据类型和大小
  auto column_id = column_name_id_[column_name];
  *column_data_type = column_data_type_[column_id];
  *column_data_type_size = ColumnDataTypeSize[*column_data_type];
  *column_shape = column_shape_[column_id];

  // Retrieve value from json从json检索值
  if (column_category == ColumnInRaw) {
    RETURN_IF_NOT_OK(GetColumnFromJson(column_name, columns_json, data_ptr, n_bytes));
    *data = reinterpret_cast<const unsigned char *>(data_ptr->get());
    return Status::OK();
  }

  // Retrieve value from blob从blob检索值
  RETURN_IF_NOT_OK(GetColumnFromBlob(column_name, columns_blob, data, data_ptr, n_bytes));
  if (*data == nullptr) {
    *data = reinterpret_cast<const unsigned char *>(data_ptr->get());
  }
  return Status::OK();
}
//引用ShardColumn空间中GetColumnFromJson函数
Status ShardColumn::GetColumnFromJson(const std::string &column_name, const json &columns_json,
                                      std::unique_ptr<unsigned char[]> *data_ptr, uint64_t *n_bytes) {
  RETURN_UNEXPECTED_IF_NULL(n_bytes);
  RETURN_UNEXPECTED_IF_NULL(data_ptr);
  auto column_id = column_name_id_[column_name];
  auto column_data_type = column_data_type_[column_id];

  // Initialize num bytes初始化（以字节为单位）
  *n_bytes = ColumnDataTypeSize[column_data_type];
  auto json_column_value = columns_json[column_name];
  CHECK_FAIL_RETURN_UNEXPECTED(json_column_value.is_string() || json_column_value.is_number(),
                               "[Internal ERROR] the value of column: " + column_name +
                                 " should be string or number but got: " + json_column_value.dump());
  //通过column_data_type选择操作方法
  switch (column_data_type) {
    case ColumnFloat32: {
      return GetFloat<float>(data_ptr, json_column_value, false);
    }
    case ColumnFloat64: {
      return GetFloat<double>(data_ptr, json_column_value, true);
    }
    case ColumnInt32: {
      return GetInt<int32_t>(data_ptr, json_column_value);
    }
    case ColumnInt64: {
      return GetInt<int64_t>(data_ptr, json_column_value);
    }
    default: {
      // Convert string to c_str将字符串转换为c_str
      std::string tmp_string;
      if (json_column_value.is_string()) {
        tmp_string = json_column_value.get<string>();
      } else {
        tmp_string = json_column_value.dump();
      }
      *n_bytes = tmp_string.size();
      auto data = reinterpret_cast<const unsigned char *>(common::SafeCStr(tmp_string));
      *data_ptr = std::make_unique<unsigned char[]>(*n_bytes);
      for (uint32_t i = 0; i < *n_bytes; i++) {
        (*data_ptr)[i] = *(data + i);
      }
      break;
    }
  }
  return Status::OK();
}
//创建函数模板，引用ShardColumn空间中GetFloat函数
template <typename T>
Status ShardColumn::GetFloat(std::unique_ptr<unsigned char[]> *data_ptr, const json &json_column_value,
                             bool use_double) {
  RETURN_UNEXPECTED_IF_NULL(data_ptr);
  std::unique_ptr<T[]> array_data = std::make_unique<T[]>(1);
  if (json_column_value.is_number()) {
    array_data[0] = json_column_value;
  } else {
    // Convert string to float将字符串转换为浮点
    try {
      if (use_double) {
        array_data[0] = json_column_value.get<double>();
      } else {
        array_data[0] = json_column_value.get<float>();
      }
    } catch (json::exception &e) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] Failed to convert column value:" + json_column_value.dump() +
                               " to type float, " + std::string(e.what()));
    }
  }
  //创建data_ptr列表
  auto data = reinterpret_cast<const unsigned char *>(array_data.get());
  *data_ptr = std::make_unique<unsigned char[]>(sizeof(T));
  for (uint32_t i = 0; i < sizeof(T); i++) {
    (*data_ptr)[i] = *(data + i);
  }
  return Status::OK();
}
//创建函数模板，引用ShardColumn空间中GetInt函数
template <typename T>
Status ShardColumn::GetInt(std::unique_ptr<unsigned char[]> *data_ptr, const json &json_column_value) {
  RETURN_UNEXPECTED_IF_NULL(data_ptr);
  std::unique_ptr<T[]> array_data = std::make_unique<T[]>(1);
  int64_t temp_value;
  bool less_than_zero = false;
  //判断json_column_value是否为integer类型，若是则令json_zero为0、temp_value等于json_column_value,并判断json_column_value是否小于0，若是则令less_than_zero为true
  //判断json_column_value是否为string类型，若是则令string_value为json_column_value
  if (json_column_value.is_number_integer()) {
    const json json_zero = 0;
    if (json_column_value < json_zero) {
      less_than_zero = true;
    }
    temp_value = json_column_value;
  } else if (json_column_value.is_string()) {
    std::string string_value = json_column_value;
    //设置异常捕捉器
    try {
      //判断string_value是否为空且string_value列表的第一位是否为‘-’,若是则给temp_value和less_than_zero赋值
      //若不是则直接给temp_value赋值
      if (!string_value.empty() && string_value[0] == '-') {
        temp_value = std::stoll(string_value);
        less_than_zero = true;
      } else {
        temp_value = static_cast<int64_t>(std::stoull(string_value));
      }
    } catch (std::invalid_argument &e) {//若问题类型为参数无效，则返回相应错误信息
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] Failed to convert column value:" + string_value + " to type int, " +
                               std::string(e.what()));
    } catch (std::out_of_range &e) {//若问题类型为超出范围，则返回相应错误信息
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] Failed to convert column value:" + string_value + " to type int, " +
                               std::string(e.what()));
    }
  } else {//若均不符合，则返回相应错误信息
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] column value should be type string or number but got: " +
                             json_column_value.dump());
  }
  //进行判断，若符合则返回相应错误信息
  if ((less_than_zero && temp_value < static_cast<int64_t>(std::numeric_limits<T>::min())) ||
      (!less_than_zero && static_cast<uint64_t>(temp_value) > static_cast<uint64_t>(std::numeric_limits<T>::max()))) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] column value: " + std::to_string(temp_value) + " is out of range.");
  }
  array_data[0] = static_cast<T>(temp_value);
  //进入循环，建立data_ptr列表
  auto data = reinterpret_cast<const unsigned char *>(array_data.get());
  *data_ptr = std::make_unique<unsigned char[]>(sizeof(T));
  for (uint32_t i = 0; i < sizeof(T); i++) {
    (*data_ptr)[i] = *(data + i);
  }
  return Status::OK();
}
//引用ShardColumn空间中GetColumnFromBlob函数
//判断已建立的column_name_id_列表和column_data_type_列表是否相同并进行操作
Status ShardColumn::GetColumnFromBlob(const std::string &column_name, const std::vector<uint8_t> &columns_blob,
                                      const unsigned char **data, std::unique_ptr<unsigned char[]> *data_ptr,
                                      uint64_t *const n_bytes) {
  RETURN_UNEXPECTED_IF_NULL(data);
  uint64_t offset_address = 0;
  auto column_id = column_name_id_[column_name];
  RETURN_IF_NOT_OK(GetColumnAddressInBlock(column_id, columns_blob, n_bytes, &offset_address));
  auto column_data_type = column_data_type_[column_id];
  if (has_compress_blob_ && column_data_type == ColumnInt32) {
    RETURN_IF_NOT_OK(UncompressInt<int32_t>(column_id, data_ptr, columns_blob, n_bytes, offset_address));
  } else if (has_compress_blob_ && column_data_type == ColumnInt64) {
    RETURN_IF_NOT_OK(UncompressInt<int64_t>(column_id, data_ptr, columns_blob, n_bytes, offset_address));
  } else {
    *data = reinterpret_cast<const unsigned char *>(&(columns_blob[offset_address]));
  }

  return Status::OK();
}
//引用ShardColumn空间中GetColumnName函数
//判断it_column是佛偶为column_name_id_列表的结尾，若是则返回ColumnNotFound
//若不是则返回ColumnInRaw或ColumnInBlob
ColumnCategory ShardColumn::CheckColumnName(const std::string &column_name) {
  auto it_column = column_name_id_.find(column_name);
  if (it_column == column_name_id_.end()) {
    return ColumnNotFound;
  }
  auto it_blob = blob_column_id_.find(column_name);
  return it_blob == blob_column_id_.end() ? ColumnInRaw : ColumnInBlob;
}
//引用ShardColumn空间中CompressBlob函数
std::vector<uint8_t> ShardColumn::CompressBlob(const std::vector<uint8_t> &blob, int64_t *compression_size) {
  // Skip if no compress columns如果没有压缩列，则跳过
  *compression_size = 0;
  if (!CheckCompressBlob()) {
    return blob;
  }

  std::vector<uint8_t> dst_blob;
  uint64_t i_src = 0;
  for (int64_t i = 0; i < num_blob_column_; i++) {
    // Get column data type获取列数据类型
    auto src_data_type = column_data_type_[column_name_id_[blob_column_[i]]];
    auto int_type = src_data_type == ColumnInt32 ? kInt32Type : kInt64Type;

    // Compress and return is blob has 1 column only压缩并返回blob只有1列
    if (num_blob_column_ == 1) {
      dst_blob = CompressInt(blob, int_type);
      *compression_size = static_cast<int64_t>(blob.size()) - static_cast<int64_t>(dst_blob.size());
      return dst_blob;
    }

    // Just copy and continue if column dat type is not int32/int64如果列数据类型不是int32/int64，只需复制并继续
    uint64_t num_bytes = BytesBigToUInt64(blob, i_src, kInt64Type);
    if (src_data_type != ColumnInt32 && src_data_type != ColumnInt64) {
      dst_blob.insert(dst_blob.end(), blob.begin() + i_src, blob.begin() + i_src + kInt64Len + num_bytes);
      i_src += kInt64Len + num_bytes;
      continue;
    }

    // Get column slice in source blob获取源blob中的列切片
    std::vector<uint8_t> blob_slice(blob.begin() + i_src + kInt64Len, blob.begin() + i_src + kInt64Len + num_bytes);
    // Compress column压缩列
    auto dst_blob_slice = CompressInt(blob_slice, int_type);
    // Get new column size获取新列大小
    auto new_blob_size = UIntToBytesBig(dst_blob_slice.size(), kInt64Type);
    // Append new column size附加新列大小
    dst_blob.insert(dst_blob.end(), new_blob_size.begin(), new_blob_size.end());
    // Append new column data附加新列数据
    dst_blob.insert(dst_blob.end(), dst_blob_slice.begin(), dst_blob_slice.end());
    i_src += kInt64Len + num_bytes;
  }
  MS_LOG(DEBUG) << "Compress blob data from " << blob.size() << " to " << dst_blob.size() << ".";
  *compression_size = static_cast<int64_t>(blob.size()) - static_cast<int64_t>(dst_blob.size());
  return dst_blob;
}
//引用ShardColumn空间中CompressBlob函数
vector<uint8_t> ShardColumn::CompressInt(const vector<uint8_t> &src_bytes, const IntegerType &int_type) {
  uint64_t i_size = kUnsignedOne << static_cast<uint8_t>(int_type);
  // Get number of elements获取元素数
  uint64_t src_n_int = src_bytes.size() / i_size;
  // Calculate bitmap size (bytes)计算位图大小（字节）
  uint64_t bitmap_size = (src_n_int + kNumDataOfByte - 1) / kNumDataOfByte;

  // Initialize destination blob, more space than needed, will be resized初始化目标blob，超出所需空间，将调整大小
  vector<uint8_t> dst_bytes(kBytesOfColumnLen + bitmap_size + src_bytes.size(), 0);

  // Write number of elements to destination blob将元素数写入目标blob
  vector<uint8_t> size_by_bytes = UIntToBytesBig(src_n_int, kInt32Type);
  for (uint64_t n = 0; n < kBytesOfColumnLen; n++) {
    dst_bytes[n] = size_by_bytes[n];
  }

  // Write compressed int写入压缩int
  uint64_t i_dst = kBytesOfColumnLen + bitmap_size;
  for (uint64_t i = 0; i < src_n_int; i++) {
    // Initialize destination data type初始化目标数据类型
    IntegerType dst_int_type = kInt8Type;
    // Shift to next int position移到下一个int位置
    uint64_t pos = i * (kUnsignedOne << static_cast<uint8_t>(int_type));
    // Narrow down this int缩小这个整数
    int64_t i_n = BytesLittleToMinIntType(src_bytes, pos, int_type, &dst_int_type);

    // Write this int to destination blob将此int写入目标blob
    uint64_t u_n = *reinterpret_cast<uint64_t *>(&i_n);
    auto temp_bytes = UIntToBytesLittle(u_n, dst_int_type);
    for (uint64_t j = 0; j < (kUnsignedOne << static_cast<uint8_t>(dst_int_type)); j++) {
      dst_bytes[i_dst++] = temp_bytes[j];
    }

    // Update date type in bit map更新位图中的日期类型
    dst_bytes[i / kNumDataOfByte + kBytesOfColumnLen] |=
      (static_cast<uint8_t>(dst_int_type) << (kDataTypeBits * (kNumDataOfByte - kUnsignedOne - (i % kNumDataOfByte))));
  }
  // Resize destination blob调整目标blob的大小
  dst_bytes.resize(i_dst);
  MS_LOG(DEBUG) << "Compress blob field from " << src_bytes.size() << " to " << dst_bytes.size() << ".";
  return dst_bytes;
}
//引用ShardColumn空间中GetColumnAddressInBlock函数
Status ShardColumn::GetColumnAddressInBlock(const uint64_t &column_id, const std::vector<uint8_t> &columns_blob,
                                            uint64_t *num_bytes, uint64_t *shift_idx) {
  RETURN_UNEXPECTED_IF_NULL(num_bytes);
  RETURN_UNEXPECTED_IF_NULL(shift_idx);
  //判断num_blob_column_是否为1，若是则用指针记录columns_blob列表的大小并返回
  if (num_blob_column_ == 1) {
    *num_bytes = columns_blob.size();
    *shift_idx = 0;
    return Status::OK();
  }
  auto blob_id = blob_column_id_[column_name_[column_id]];
  //进入循环，按步骤调用BytesBigToUInt64
  for (int32_t i = 0; i < blob_id; i++) {
    *shift_idx += kInt64Len + BytesBigToUInt64(columns_blob, *shift_idx, kInt64Type);
  }
  *num_bytes = BytesBigToUInt64(columns_blob, *shift_idx, kInt64Type);

  (*shift_idx) += kInt64Len;

  return Status::OK();
}
//创建函数模板，引用ShardColumn空间中UncompressInt函数
template <typename T>
Status ShardColumn::UncompressInt(const uint64_t &column_id, std::unique_ptr<unsigned char[]> *const data_ptr,
                                  const std::vector<uint8_t> &columns_blob, uint64_t *num_bytes, uint64_t shift_idx) {
  RETURN_UNEXPECTED_IF_NULL(data_ptr);
  RETURN_UNEXPECTED_IF_NULL(num_bytes);
  auto num_elements = BytesBigToUInt64(columns_blob, shift_idx, kInt32Type);
  *num_bytes = sizeof(T) * num_elements;

  // Parse integer array解析整数数组
  uint64_t i_source = shift_idx + kBytesOfColumnLen + (num_elements + kNumDataOfByte - 1) / kNumDataOfByte;
  auto array_data = std::make_unique<T[]>(num_elements);

  for (uint64_t i = 0; i < num_elements; i++) {
    uint8_t iBitMap = columns_blob[shift_idx + kBytesOfColumnLen + i / kNumDataOfByte];
    uint64_t i_type = (iBitMap >> ((kNumDataOfByte - 1 - (i % kNumDataOfByte)) * kDataTypeBits)) & kDataTypeBitMask;
    auto mr_int_type = static_cast<IntegerType>(i_type);
    int64_t i64 = BytesLittleToMinIntType(columns_blob, i_source, mr_int_type);
    i_source += (kUnsignedOne << i_type);
    array_data[i] = static_cast<T>(i64);
  }

  auto data = reinterpret_cast<const unsigned char *>(array_data.get());
  *data_ptr = std::make_unique<unsigned char[]>(*num_bytes);
  // field is none. for example: numpy is null字段为无。例如：numpy为null
  if (*num_bytes == 0) {
    return Status::OK();
  }
  CHECK_FAIL_RETURN_UNEXPECTED(memcpy_s(data_ptr->get(), *num_bytes, data, *num_bytes) == 0,
                               "[Internal ERROR] Failed to call securec func [memcpy_s]");
  return Status::OK();
}
//引用ShardColumn空间中BytesBigToUInt64函数
//进入循环，计算result的值并返回
uint64_t ShardColumn::BytesBigToUInt64(const std::vector<uint8_t> &bytes_array, const uint64_t &pos,
                                       const IntegerType &i_type) {
  uint64_t result = 0;
  for (uint64_t i = 0; i < (kUnsignedOne << static_cast<uint8_t>(i_type)); i++) {
    result = (result << kBitsOfByte) + bytes_array[pos + i];
  }
  return result;
}
//引用ShardColumn空间中UIntToBytesBig函数
//进入循环，根据操作计算result的值并返回
std::vector<uint8_t> ShardColumn::UIntToBytesBig(uint64_t value, const IntegerType &i_type) {
  uint64_t n_bytes = kUnsignedOne << static_cast<uint8_t>(i_type);
  std::vector<uint8_t> result(n_bytes, 0);
  for (uint64_t i = 0; i < n_bytes; i++) {
    result[n_bytes - 1 - i] = value & std::numeric_limits<uint8_t>::max();
    value >>= kBitsOfByte;
  }
  return result;
}
//引用ShardColumn空间中UIntToBytesLittle函数
//进入循环，根据操作计算result的值并返回
std::vector<uint8_t> ShardColumn::UIntToBytesLittle(uint64_t value, const IntegerType &i_type) {
  uint64_t n_bytes = kUnsignedOne << static_cast<uint8_t>(i_type);
  std::vector<uint8_t> result(n_bytes, 0);
  for (uint64_t i = 0; i < n_bytes; i++) {
    result[i] = value & std::numeric_limits<uint8_t>::max();
    value >>= kBitsOfByte;
  }
  return result;
}
//引用ShardColumn空间中BytesLittleToMinIntType函数
int64_t ShardColumn::BytesLittleToMinIntType(const std::vector<uint8_t> &bytes_array, const uint64_t &pos,
                                             const IntegerType &src_i_type, IntegerType *dst_i_type) {
  uint64_t u_temp = 0;
  //进入循环，计算u_temp的值
  for (uint64_t i = 0; i < (kUnsignedOne << static_cast<uint8_t>(src_i_type)); i++) {
    u_temp = (u_temp << kBitsOfByte) +
             bytes_array[pos + (kUnsignedOne << static_cast<uint8_t>(src_i_type)) - kUnsignedOne - i];
  }
  //根据src_i_type的类型判断，并选择相应的处理
  int64_t i_out;
  switch (src_i_type) {
    case kInt8Type: {
      i_out = (int8_t)(u_temp & std::numeric_limits<uint8_t>::max());
      break;
    }
    case kInt16Type: {
      i_out = (int16_t)(u_temp & std::numeric_limits<uint16_t>::max());
      break;
    }
    case kInt32Type: {
      i_out = (int32_t)(u_temp & std::numeric_limits<uint32_t>::max());
      break;
    }
    case kInt64Type: {
      i_out = (int64_t)(u_temp & std::numeric_limits<uint64_t>::max());
      break;
    }
    default: {
      i_out = 0;
    }
  }
  //判断dst_i_type是否为假，若是则直接返回i_out
  if (!dst_i_type) {
    return i_out;
  }
  //判断i_out的取值，给指针dst_i_type赋值
  if (i_out >= static_cast<int64_t>(std::numeric_limits<int8_t>::min()) &&
      i_out <= static_cast<int64_t>(std::numeric_limits<int8_t>::max())) {
    *dst_i_type = kInt8Type;
  } else if (i_out >= static_cast<int64_t>(std::numeric_limits<int16_t>::min()) &&
             i_out <= static_cast<int64_t>(std::numeric_limits<int16_t>::max())) {
    *dst_i_type = kInt16Type;
  } else if (i_out >= static_cast<int64_t>(std::numeric_limits<int32_t>::min()) &&
             i_out <= static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
    *dst_i_type = kInt32Type;
  } else {
    *dst_i_type = kInt64Type;
  }
  return i_out;
}
}  // namespace mindrecord
}  // namespace mindspore
