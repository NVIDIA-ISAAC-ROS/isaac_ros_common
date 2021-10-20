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
#ifndef NVIDIA_GXF_CORE_FACTORY_HPP
#define NVIDIA_GXF_CORE_FACTORY_HPP

#include <memory>
#include <utility>

#include "gxf/core/expected.hpp"
#include "gxf/std/default_extension.hpp"

// Helper macro to create an extensions factory. Every component in this extensions must be
// explicitly registered. Otherwise it can not be used by GXF applications.
//
// A component can be registered using the macro GXF_EXT_FACTORY_ADD. For each
// component the base class need to be specified. Components base classes must be registered before
// they can be used as a base class in a component registration. If a component does not have a base
// class the macro GXF_EXT_FACTORY_ADD_0 is used instead.
//
// Components can have at most one base class. Multiple base classes are not supported.
//
// Every component must be registered with a unique 128-bit identifier. The identifer must be
// unique across all existing extensions.
//
// Note that the extension factory can also be created manually without using these macros.
//
// Usage example:
//   GXF_EXT_FACTORY_BEGIN(0x8ec2d5d6b5df48bf, 0x8dee0252606fdd7e, "1.0.0")
//   GXF_EXT_FACTORY_SET_INFO("Extension Name", "Extension Desc", "Author", "LICENSE")
//   GXF_EXT_FACTORY_ADD(0x792151bf31384603, 0xa9125ca91828dea8,
//                       nvidia::gxf::Queue, nvidia::gxf::Component,
//                       "Interface for storing entities in a queue");
//   GXF_EXT_FACTORY_ADD(0xc30cc60f0db2409d, 0x92b6b2db92e02cce,
//                        nvidia::gxf::Transmitter, nvidia::gxf::Queue,
//                        "Interface for publishing entities");
//   ...
//   GXF_EXT_FACTORY_END()
#define GXF_EXT_FACTORY_BEGIN()                                                                    \
  namespace {                                                                                      \
    nvidia::gxf::Expected<std::unique_ptr<nvidia::gxf::ComponentFactory>>                          \
    CreateComponentFactory() {                                                                     \
      auto factory = std::make_unique<nvidia::gxf::DefaultExtension>();                            \
      if (!factory) { return nvidia::gxf::Unexpected{GXF_OUT_OF_MEMORY}; }                         \

// See GXF_EXT_FACTORY_BEGIN for more information.
#define GXF_EXT_FACTORY_SET_INFO(H1, H2, NAME, DESC, AUTHOR, VERSION, LICENSE)                     \
      {                                                                                            \
        const nvidia::gxf::Expected<void> result = factory->setInfo({(H1), (H2)}, NAME, DESC,      \
                                                                     AUTHOR, VERSION, LICENSE);    \
        if (!result) { return nvidia::gxf::ForwardError(result); }                                 \
      }                                                                                            \

// See GXF_EXT_FACTORY_BEGIN for more information.
#define GXF_EXT_FACTORY_ADD_0(H1, H2, TYPE, DESC)                                                  \
      {                                                                                            \
        const nvidia::gxf::Expected<void> result = factory->add<TYPE>({(H1), (H2)}, DESC);         \
        if (!result) { return nvidia::gxf::ForwardError(result); }                                 \
      }                                                                                            \

// See GXF_EXT_FACTORY_BEGIN for more information.
#define GXF_EXT_FACTORY_ADD_0_LITE(H1, H2, TYPE)                                                   \
      {                                                                                            \
        const nvidia::gxf::Expected<void> result = factory->add<TYPE>({(H1), (H2)}, "");           \
        if (!result) { return nvidia::gxf::ForwardError(result); }                                 \
      }                                                                                            \

// See GXF_EXT_FACTORY_BEGIN for more information.
#define GXF_EXT_FACTORY_ADD(H1, H2, TYPE, BASE, DESC)                                              \
      {                                                                                            \
        const nvidia::gxf::Expected<void> result = factory->add<TYPE, BASE>({(H1), (H2)}, DESC);   \
        if (!result) { return nvidia::gxf::ForwardError(result); }                                 \
      }                                                                                            \

// See GXF_EXT_FACTORY_BEGIN for more information.
#define GXF_EXT_FACTORY_ADD_LITE(H1, H2, TYPE, BASE)                                               \
      {                                                                                            \
        const nvidia::gxf::Expected<void> result = factory->add<TYPE, BASE>({(H1), (H2)}, "");     \
        if (!result) { return nvidia::gxf::ForwardError(result); }                                 \
      }                                                                                            \

// See GXF_EXT_FACTORY_BEGIN for more information.
#define GXF_EXT_FACTORY_END()                                                                      \
      const nvidia::gxf::Expected<void> result = factory->checkInfo();                             \
      if (!result) { return nvidia::gxf::ForwardError(result); }                                   \
      return std::move(factory);                                                                   \
    }                                                                                              \
  }  /* namespace */                                                                               \
                                                                                                   \
  extern "C" {                                                                                     \
    gxf_result_t GxfExtensionFactory(void** result) {                                              \
      static nvidia::gxf::Expected<std::unique_ptr<nvidia::gxf::ComponentFactory>> s_factory       \
          = CreateComponentFactory();                                                              \
      if (!s_factory) { return s_factory.error(); }                                                \
      *result = s_factory.value().get();                                                           \
      return GXF_SUCCESS;                                                                          \
    }                                                                                              \
  }  /* extern "C" */                                                                              \

#endif
