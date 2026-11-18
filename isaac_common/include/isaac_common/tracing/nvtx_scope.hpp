// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ISAAC_COMMON__TRACING__NVTX_SCOPE_HPP_
#define ISAAC_COMMON__TRACING__NVTX_SCOPE_HPP_

/// Scoped NVTX ranges for the deploy pipeline, in the `isaac_deploy` domain. nvtx_trace_bridge.hpp
/// covers what happens inside Triton; this covers everything around it.

// Unconditional: the disabled-build stubs take uint64_t, which nvToolsExt.h would otherwise supply.
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(USE_NVTX)
#include <sys/syscall.h>
#include <unistd.h>

#include <nvtx3/nvToolsExt.h>
#endif

namespace nvidia::isaac_ros::tracing
{

/// Palette for pipeline stages, kept distinct from the Triton bridge's so the domains separate.
enum Color : unsigned int
{
  kColorUpdate = 0xFF6A5ACD,        ///< control-cycle envelope - slate blue
  kColorReadState = 0xFF3CB371,     ///< hardware state ingest  - medium sea green
  kColorTopicInput = 0xFF20B2AA,    ///< topic ingest           - light sea green
  kColorStateBuilder = 0xFFDAA520,  ///< NN input assembly      - goldenrod
  kColorRunner = 0xFF4682B4,        ///< inference dispatch     - steel blue
  kColorActionChunk = 0xFFCD5C5C,   ///< NN output decode       - indian red
  kColorWriteCommand = 0xFFFF8C00,  ///< command interface emit - dark orange
  kColorSafety = 0xFFB22222,        ///< safety controller      - firebrick
  kColorBuildRequest = 0xFF9370DB,  ///< Triton request marshal - medium purple
  kColorWaitResponse = 0xFF008B8B,  ///< blocked on the future  - dark cyan
  kColorDecodeOutput = 0xFFBDB76B,  ///< response -> torch copy - dark khaki
  kColorWaitRelease = 0xFF778899,   ///< release barrier        - light slate gray
  kColorInputTerm = 0xFF8FBC8F,     ///< one input term         - dark sea green
  kColorOutputTerm = 0xFFD2691E,    ///< one output term        - chocolate
};

/// The single gate for the whole pipeline; nvtx_trace_bridge.hpp forwards to this. Follows the
/// collector (NVTX_INJECTION64_PATH, inherited across fork/exec); ISAAC_DEPLOY_NVTX overrides.
inline bool enabled()
{
#if defined(USE_NVTX)
  static const bool on = [] {
      const char * v = std::getenv("ISAAC_DEPLOY_NVTX");
      if (v != nullptr) {
        return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
               std::strcmp(v, "on") == 0 || std::strcmp(v, "yes") == 0;
      }
      return std::getenv("NVTX_INJECTION64_PATH") != nullptr;
    }();
  return on;
#else
  return false;
#endif
}

#if defined(USE_NVTX)

inline nvtxDomainHandle_t domain()
{
  static const nvtxDomainHandle_t handle = nvtxDomainCreateA("isaac_ros");
  return handle;
}

/// A name registered with the domain. Construct once per call site (function-local static).
class RegisteredName
{
public:
  explicit RegisteredName(const char * name)
  : handle_(enabled() ? nvtxDomainRegisterStringA(domain(), name) : nullptr) {}

  nvtxStringHandle_t handle() const {return handle_;}

private:
  nvtxStringHandle_t handle_;
};

/// Push/pop is thread-scoped, so a range must open and close on the same thread.
class ScopedRange
{
public:
  ScopedRange(const RegisteredName & name, unsigned int color)
  : active_(enabled() && name.handle() != nullptr)
  {
    if (!active_) {
      return;
    }
    nvtxEventAttributes_t attr{};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = color;
    attr.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    attr.message.registered = name.handle();
    nvtxDomainRangePushEx(domain(), &attr);
  }

  /// Variant carrying the uint64 correlation key. NVTX fixes attributes at push, so pass it here.
  ScopedRange(const RegisteredName & name, unsigned int color, uint64_t payload)
  : active_(enabled() && name.handle() != nullptr)
  {
    if (!active_) {
      return;
    }
    nvtxEventAttributes_t attr{};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = color;
    attr.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    attr.message.registered = name.handle();
    attr.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
    attr.payload.ullValue = payload;
    nvtxDomainRangePushEx(domain(), &attr);
  }

  /// Variant for runtime names. Costs an ASCII copy per event; prefer RegisteredName elsewhere.
  ScopedRange(const char * name, unsigned int color)
  : active_(enabled() && name != nullptr)
  {
    if (!active_) {
      return;
    }
    nvtxEventAttributes_t attr{};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = color;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    nvtxDomainRangePushEx(domain(), &attr);
  }

  ~ScopedRange() {end();}

  /// Closes early, for a phase ending partway through a scope. Idempotent; the dtor is a backstop.
  void end()
  {
    if (active_) {
      nvtxDomainRangePop(domain());
      active_ = false;
    }
  }

  ScopedRange(const ScopedRange &) = delete;
  ScopedRange & operator=(const ScopedRange &) = delete;

private:
  bool active_;
};

/// Mark carrying a uint64 payload, for values a range payload cannot take (fixed at push time).
inline void mark_u64(const char * name, uint64_t value)
{
  if (!enabled() || name == nullptr) {
    return;
  }
  nvtxEventAttributes_t attr{};
  attr.version = NVTX_VERSION;
  attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
  attr.message.ascii = name;
  attr.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
  attr.payload.ullValue = value;
  nvtxDomainMarkEx(domain(), &attr);
}

/// Registered-handle form, for the control-rate call sites whose names are fixed at compile time.
inline void mark_u64(const RegisteredName & name, uint64_t value)
{
  if (!enabled() || name.handle() == nullptr) {
    return;
  }
  nvtxEventAttributes_t attr{};
  attr.version = NVTX_VERSION;
  attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  attr.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
  attr.message.registered = name.handle();
  attr.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
  attr.payload.ullValue = value;
  nvtxDomainMarkEx(domain(), &attr);
}

/// Labels the calling OS thread so the timeline reads a role. SYS_gettid: nsys keys on kernel TID.
inline void name_thread(const char * name)
{
  thread_local bool named = false;
  if (named || !enabled() || name == nullptr) {
    return;
  }
  named = true;
  nvtxNameOsThreadA(static_cast<uint32_t>(::syscall(SYS_gettid)), name);
}

#else  // USE_NVTX

/// No-op stand-ins so instrumented call sites compile unchanged without the trace bridge.
class RegisteredName
{
public:
  explicit RegisteredName(const char *) {}
};

class ScopedRange
{
public:
  ScopedRange(const RegisteredName &, unsigned int) {}
  ScopedRange(const RegisteredName &, unsigned int, uint64_t) {}
  ScopedRange(const char *, unsigned int) {}
  void end() {}
  ScopedRange(const ScopedRange &) = delete;
  ScopedRange & operator=(const ScopedRange &) = delete;
};

inline void mark_u64(const char *, uint64_t) {}
inline void mark_u64(const RegisteredName &, uint64_t) {}
inline void name_thread(const char *) {}

#endif  // USE_NVTX

}  // namespace nvidia::isaac_ros::tracing

/// Range covering the rest of the scope. One per scope; for staged functions use ScopedRange.
#define ISAAC_DEPLOY_NVTX_SCOPE(name, color) \
  static const ::nvidia::isaac_ros::tracing::RegisteredName _isaac_nvtx_name{name}; \
  const ::nvidia::isaac_ros::tracing::ScopedRange _isaac_nvtx_scope{_isaac_nvtx_name, color}

#endif  // ISAAC_COMMON__TRACING__NVTX_SCOPE_HPP_
