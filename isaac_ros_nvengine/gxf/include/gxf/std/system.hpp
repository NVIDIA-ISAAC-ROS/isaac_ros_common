/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef NVIDIA_GXF_CORE_SYSTEM_HPP
#define NVIDIA_GXF_CORE_SYSTEM_HPP

#include "gxf/core/component.hpp"

namespace nvidia {
namespace gxf {

// Component interface for systems which are run as part of the application run cycle.
class System : public Component {
 public:
  virtual ~System() = default;

  virtual gxf_result_t offer_abi(gxf_uid_t eid) = 0;
  virtual gxf_result_t runAsync_abi() = 0;
  virtual gxf_result_t stop_abi() = 0;
  virtual gxf_result_t wait_abi() = 0;
  virtual gxf_result_t event_notify_abi(gxf_uid_t eid) = 0;

  // These apis will be made pure virtual interfaces post V2.3 to replace offer_abi() interface
  virtual gxf_result_t schedule_abi(gxf_uid_t eid) { return GXF_SUCCESS; }
  virtual gxf_result_t unschedule_abi(gxf_uid_t eid) { return GXF_SUCCESS; }

  Expected<void> offer(const Entity& entity);
  Expected<void> schedule(const Entity& entity);
  Expected<void> unschedule(const Entity& entity);
  Expected<void> runAsync();
  Expected<void> stop();
  Expected<void> wait();
  Expected<void> event_notify(gxf_uid_t eid);
};

}  // namespace gxf
}  // namespace nvidia

#endif
