// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include "isaac_common/tracing/nvtx_scope.hpp"

namespace nvidia::isaac_ros::tracing
{

// Compile the complete helper surface in both USE_NVTX and stub configurations.
TEST(NvtxScopeTest, InitsScopeInBothConfigurations)
{
  RegisteredName name("test_scope");

  {
    ScopedRange range(name, kColorUpdate);
    ScopedRange payload_range(name, kColorReadState, uint64_t{42});
    ScopedRange runtime_named_range("runtime_named_scope", kColorTopicInput);
    range.end();
  }

  mark_u64("test_mark", uint64_t{7});
  mark_u64(name, uint64_t{7});
  name_thread("test_thread");

  SUCCEED();
}

TEST(NvtxScopeTest, ScopeMacroCompilesAndRuns)
{
  {
    ISAAC_DEPLOY_NVTX_SCOPE("test_macro_scope", kColorRunner);
  }
  SUCCEED();
}

}  // namespace nvidia::isaac_ros::tracing
