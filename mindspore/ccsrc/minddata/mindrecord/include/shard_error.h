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

#ifndef MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_ERROR_H_
#define MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_ERROR_H_

#include <map>
#include <string>
#include "include/api/status.h"

namespace mindspore {  //命名mindspore空间
namespace mindrecord {  //命名mindrecord空间
#define RETURN_IF_NOT_OK(_s) \ //参数_s
  do {                       \
    Status __rc = (_s);      \ //将_s赋给_rc
    if (__rc.IsError()) {    \ //如果_rc.IsError()返回值为1，则返回_rc
      return __rc;           \
    }                        \
  } while (false)              //停止循环

#define RELEASE_AND_RETURN_IF_NOT_OK(_s, _db, _in) \ //参数_s,_db,_in
  do {                                             \
    Status __rc = (_s);                            \ //将_s赋给_rc
    if (__rc.IsError()) {                          \ //当_rc.IsError()返回值为1时，如果_db不为空，关闭_db数据库链接
      if ((_db) != nullptr) {                      \ 
        sqlite3_close(_db);                        \
      }                                            \
      (_in).close();                               \ //关闭_in
      return __rc;                                 \ //返回_rc
    }                                              \
  } while (false)                                    //停止循环

#define CHECK_FAIL_RETURN_UNEXPECTED(_condition, _e)                         \ //参数_condition,_e
  do {                                                                       \ 
    if (!(_condition)) {                                                     \ //如果_condition为0，返回StatusCOde下的kMDUnexpectedError, __LINE__, __FILE__, _e
      return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, _e); \
    }                                                                        \
  } while (false)                                                              //停止循环

#define RETURN_UNEXPECTED_IF_NULL(_ptr)                                         \ //参数_ptr
  do {                                                                          \
    if ((_ptr) == nullptr) {                                                    \ //如果_ptr为空指针，则err_msg为The pointer[" + std::string(#_ptr) + "] is null
      std::string err_msg = "The pointer[" + std::string(#_ptr) + "] is null."; \
      RETURN_STATUS_UNEXPECTED(err_msg);                                        \ //将err_msg传入RETURN_STATUS_UNEXPECTED()函数，返回StatusCOde下的kMDUnexpectedError, __LINE__, __FILE__,err_msg
    }                                                                           \
  } while (false)                                                                 //停止循环

#define RETURN_STATUS_UNEXPECTED(_e)                                       \ //参数_e
  do {                                                                     \ 
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, _e); \ //返回StatusCOde下的kMDUnexpectedError, __LINE__, __FILE__, _e
  } while (false)                                                            //停止循环

enum MSRStatus { //失败
  SUCCESS = 0, 
  FAILED = 1,
};

}  // namespace mindrecord
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_MINDRECORD_INCLUDE_SHARD_ERROR_H_
