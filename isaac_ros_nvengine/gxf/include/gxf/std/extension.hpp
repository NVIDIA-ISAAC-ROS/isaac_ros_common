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
#ifndef NVIDIA_GXF_STD_EXTENSION_HPP
#define NVIDIA_GXF_STD_EXTENSION_HPP

#include "gxf/core/expected.hpp"
#include "gxf/std/component_factory.hpp"

namespace nvidia {
namespace gxf {

// Interface used for extensions. An extension class is holding information about the extension
// and allows creation of components which are provided by the extension.
class Extension : public ComponentFactory {
 public:
  virtual ~Extension() = default;

  Extension(const Extension&) = delete;
  Extension(Extension&&) = delete;
  Extension& operator=(const Extension&) = delete;
  Extension& operator=(Extension&&) = delete;

  // Capture extension metadata
  virtual gxf_result_t setInfo_abi(gxf_tid_t tid, const char* name, const char* desc,
                                   const char* author, const char* version,
                                   const char* license) = 0;

  // Check if extension metadata has been captured
  virtual gxf_result_t checkInfo_abi() = 0;

  // Gets description of the extension and list of components it provides
  virtual gxf_result_t getInfo_abi(gxf_extension_info_t* info) = 0;

  // Registers all components in the extension with the given context.
  virtual gxf_result_t registerComponents_abi(gxf_context_t context) = 0;

  // Gets a list with IDs of all types which are registered with this factory.
  virtual gxf_result_t getComponentTypes_abi(gxf_tid_t* pointer, size_t* size) = 0;

  // Gets description of specified component (No parameter information)
  virtual gxf_result_t getComponentInfo_abi(const gxf_tid_t tid, gxf_component_info_t* info) = 0;

  // Gets description of specified parameter
  virtual gxf_result_t getParameterInfo_abi(gxf_context_t context, const gxf_tid_t cid,
                       const char* key, gxf_parameter_info_t* info) = 0;

  Expected<void> registerComponents(gxf_context_t context);
  Expected<void> getComponentTypes(gxf_tid_t* pointer, size_t* size);
  Expected<void> setInfo(gxf_tid_t tid, const char* name, const char* desc,
                                   const char* author, const char* version, const char* license);
  Expected<void> checkInfo();
  Expected<void> getInfo(gxf_extension_info_t* info);
  Expected<void> getComponentInfo(const gxf_tid_t tid, gxf_component_info_t* info);
  Expected<void> getParameterInfo(gxf_context_t context, const gxf_tid_t cid, const char* key,
                                  gxf_parameter_info_t* info);

 protected:
  Extension() = default;
};

}  // namespace gxf
}  // namespace nvidia

#endif
