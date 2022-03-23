/*
Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_TEST_EXTENSIONS_TEST_PARAMETERS_HPP_
#define NVIDIA_GXF_TEST_EXTENSIONS_TEST_PARAMETERS_HPP_

#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "gxf/core/component.hpp"
#include "gxf/core/entity.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/parameter_parser_std.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {

// A simple structure used for testing ParameterParser
struct MyDouble {
  double d;
};

// Support MyDouble parameters
template <>
struct ParameterParser<MyDouble> {
  static Expected<MyDouble> Parse(gxf_context_t context, gxf_uid_t component_uid, const char* key,
                                  const YAML::Node& node, const std::string& prefix) {
    const auto maybe = ParameterParser<double>::Parse(context, component_uid, key, node, prefix);
    if (!maybe) {
      return ForwardError(maybe);
    }
    return MyDouble{maybe.value()};
  }
};

namespace test {

class LoadParameterFromYamlTest : public Component {
 public:
  gxf_result_t initialize() override {
    constexpr const char* expected_more =
        "- a: st\n  b: ee\n- c: an\n  d: en\n- e:\n    - f: figy\n      g: g";
    GXF_ASSERT_TRUE(fact_);
    GXF_ASSERT_FALSE(rumor_);
    GXF_ASSERT_EQ(forty_two_, 42);
    GXF_ASSERT_EQ(minus_one_, -1);
    GXF_ASSERT_STREQ(some_text_.get().c_str(), "hello");
    GXF_ASSERT_STREQ(more_.get().c_str(), expected_more);
    return GXF_SUCCESS;
  }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(fact_, "fact");
    result &= registrar->parameter(rumor_, "rumor");
    result &= registrar->parameter(forty_two_, "forty_two");
    result &= registrar->parameter(minus_one_, "minus_one");
    result &= registrar->parameter(some_text_, "some_text");
    result &= registrar->parameter(more_, "more");
    return ToResultCode(result);
  }

 private:
  Parameter<bool> fact_;
  Parameter<bool> rumor_;
  Parameter<int> forty_two_;
  Parameter<int> minus_one_;
  Parameter<std::string> some_text_;
  Parameter<std::string> more_;
};

class VectorParameterTest : public Component {
 public:
  gxf_result_t initialize() override {
    return GXF_SUCCESS;
  }

  gxf_result_t registerInterface(Registrar* registrar) override {
    ParameterInfo<std::vector<double>> five_64_bit_floats_info {
      "five_64_bit_floats",
      "testing float vector",
      "description for five_64_bit_floats",
      "linux_x86_64, linux_aarch64",
    };
    five_64_bit_floats_info.rank = 1;
    five_64_bit_floats_info.shape[0] = 5;
    ParameterInfo<std::vector<int64_t>> five_64_bit_ints_info {
      "five_64_bit_ints",
      "testing float vector",
      "description for five_64_bit_ints",
      "linux_x86_64, linux_aarch64",
    };
    five_64_bit_ints_info.rank = 1;
    five_64_bit_ints_info.shape[0] = 5;
    ParameterInfo<std::vector<uint64_t>> five_64_bit_unsigned_info {
      "five_64_bit_unsigned_ints",
      "testing int64 vector",
      "description for five_64_bit_unsigned_ints",
      "linux_x86_64, linux_aarch64",
    };
    five_64_bit_unsigned_info.rank = 1;
    five_64_bit_unsigned_info.shape[0] = 5;
    ParameterInfo<std::vector<int32_t>> five_32_bit_ints_info {
      "five_32_bit_ints",
      "testing float vector",
      "description for five_32_bit_ints",
      "linux_x86_64, linux_aarch64",
    };
    five_32_bit_ints_info.rank = 1;
    five_32_bit_ints_info.shape[0] = 5;
    ParameterInfo<std::vector<std::vector<double>>> six_64_bit_float_2d_info {
      "six_64_bit_float_2d",
      "testing float vector",
      "description for six_64_bit_float_2d",
      "linux_x86_64, linux_aarch64",
    };
    six_64_bit_float_2d_info.rank = 2;
    six_64_bit_float_2d_info.shape[0] = 2;
    six_64_bit_float_2d_info.shape[1] = 3;
    ParameterInfo<std::vector<std::vector<int64_t>>> six_64_bit_int_2d_info {
      "six_64_bit_int_2d",
      "testing float vector",
      "description for six_64_bit_int_2d",
      "linux_x86_64, linux_aarch64",
    };
    six_64_bit_int_2d_info.rank = 2;
    six_64_bit_int_2d_info.shape[0] = 2;
    six_64_bit_int_2d_info.shape[1] = 3;
    ParameterInfo<std::vector<std::vector<uint64_t>>> six_64_bit_uint_2d_info {
      "six_64_bit_uint_2d",
      "testing float vector",
      "description for six_64_bit_uint_2d",
      "linux_x86_64, linux_aarch64",
    };
    six_64_bit_uint_2d_info.rank = 2;
    six_64_bit_uint_2d_info.shape[0] = 2;
    six_64_bit_uint_2d_info.shape[1] = 3;
    ParameterInfo<std::vector<std::vector<int32_t>>> six_32_bit_int_2d_info {
      "six_32_bit_int_2d",
      "testing float vector",
      "description for vector_of_float",
      "linux_x86_64, linux_aarch64",
    };
    six_32_bit_int_2d_info.rank = 2;
    six_32_bit_int_2d_info.shape[0] = 2;
    six_32_bit_int_2d_info.shape[1] = 3;

    Expected<void> result;
    result &= registrar->parameter(floats_, five_64_bit_floats_info);
    result &= registrar->parameter(int64s_, five_64_bit_ints_info);
    result &= registrar->parameter(uint64s_, five_64_bit_unsigned_info);
    result &= registrar->parameter(int32s_, five_32_bit_ints_info);
    result &= registrar->parameter(floats_2d, six_64_bit_float_2d_info);
    result &= registrar->parameter(int64s_2d, six_64_bit_int_2d_info);
    result &= registrar->parameter(uint64s_2d, six_64_bit_uint_2d_info);
    result &= registrar->parameter(int32s_2d, six_32_bit_int_2d_info);
    return ToResultCode(result);
  }

 private:
  Parameter<std::vector<double>> floats_;
  Parameter<std::vector<int64_t>> int64s_;
  Parameter<std::vector<uint64_t>> uint64s_;
  Parameter<std::vector<int32_t>> int32s_;
  Parameter<std::vector<std::vector<double>>> floats_2d;
  Parameter<std::vector<std::vector<int64_t>>> int64s_2d;
  Parameter<std::vector<std::vector<uint64_t>>> uint64s_2d;
  Parameter<std::vector<std::vector<int32_t>>> int32s_2d;
};

class RegisterParameterInterfaceTest : public Component {
 public:
  gxf_result_t initialize() override {
    if (mandatory_no_default_ != 1) {
      return GXF_FAILURE;
    }
    if (mandatory_with_default_ != 2) {
      return GXF_FAILURE;
    }

    const auto result = optional_no_default_.try_get();
    if (!result || result.value() != 3) {
      return GXF_FAILURE;
    }

    const auto result2 = optional_with_default_.try_get();
    if (!result2 || result2.value() != 4) {
      return GXF_FAILURE;
    }

    return GXF_SUCCESS;
  }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(mandatory_no_default_,
                                   {"mandatory_no_default", "Mandatory No Default", "N/A"});

    ParameterInfo<int64_t> mandate_with_default_info{
        "mandatory_with_default",
        "Mandatory With Default",
        "N/A",
        "linux_x86_64, linux_aarch64",
    };
    mandate_with_default_info.value_default = 3L;
    mandate_with_default_info.value_range = {-100, 100, 1};
    result &= registrar->parameter(mandatory_with_default_, mandate_with_default_info);

    result &=
        registrar->parameter(optional_no_default_, "optional_no_default", "Mandatory No Default",
                             "N/A", Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);

    ParameterInfo<uint64_t> optional_with_default_info{
        "optional_with_default",
        "Optional With Default",
        "N/A",
    };
    optional_with_default_info.flags = GXF_PARAMETER_FLAGS_OPTIONAL,
    optional_with_default_info.value_default = 5UL;
    optional_with_default_info.value_range = {10UL, 1000UL, 10UL};
    result &= registrar->parameter(optional_with_default_, optional_with_default_info);
    result &= registrar->parameter(std_string_text_, "std_string_text", "Std_String_Text",
                                   "This is a std::string",
                                   std::string("Default value of std::string text"));
    result &= registrar->parameter(bool_default_, "bool_default", "Bool_Default",
                                   "Description of bool default", true);

    ParameterInfo<double> double_default_info{
        "double_default",
        "Double_Default",
        "Description of double default",
    };
    double_default_info.value_default = 12345.6789;
    double_default_info.value_range = {-10.0, 10.0, 1.0};
    result &= registrar->parameter(double_default_, double_default_info);

    result &= registrar->parameter(custom_parameter_, "custom_parameter");

    // Breaks build
    // std::string default_text("default char text");
    // result &= registrar->parameter(char_text_, "char_text_","Char_Text",
    //                                "This is a char array",
    //                                const_cast<char*>(default_text.c_str()));
    return ToResultCode(result);
  }

 private:
  Parameter<int64_t> mandatory_no_default_;
  Parameter<int64_t> mandatory_with_default_;
  Parameter<int64_t> optional_no_default_;
  Parameter<uint64_t> optional_with_default_;
  Parameter<std::string> std_string_text_;
  Parameter<bool> bool_default_;
  Parameter<double> double_default_;
  Parameter<std::vector<std::string>> custom_parameter_;
  Parameter<char*> char_text_;
};

class FixedVectorParameterTest : public Component {
 public:
  gxf_result_t initialize() override {
    GXF_ASSERT_EQ(fixed_vector_stack_.get().size(), kVectorSize);
    for (size_t i = 0; i < fixed_vector_stack_.get().size(); i++) {
      GXF_ASSERT_EQ(fixed_vector_stack_.get().at(i).value(), i);
    }
    return GXF_SUCCESS;
  }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(fixed_vector_stack_, "fixed_vector_stack");
    return ToResultCode(result);
  }

 private:
  static constexpr size_t kVectorSize = 16;
  Parameter<FixedVector<uint8_t, kVectorSize>> fixed_vector_stack_;
};

class StdParameterTest : public Component {
 public:
  gxf_result_t initialize() override {
    const auto& integers = integers_.get();
    GXF_ASSERT_EQ(integers.size(), 5);
    GXF_ASSERT_EQ(integers[0], 2);
    GXF_ASSERT_EQ(integers[1], 3);
    GXF_ASSERT_EQ(integers[2], 5);
    GXF_ASSERT_EQ(integers[3], 7);
    GXF_ASSERT_EQ(integers[4], 11);

    const auto& my_unsigned_int8 = my_unsigned_int8_.get();
    GXF_ASSERT_EQ(my_unsigned_int8.size(), 4);
    GXF_ASSERT_EQ(my_unsigned_int8[0], 133);
    GXF_ASSERT_EQ(my_unsigned_int8[1], 100);
    GXF_ASSERT_EQ(my_unsigned_int8[2], 1);
    GXF_ASSERT_EQ(my_unsigned_int8[3], 3);

    const auto& strings = strings_.get();
    GXF_ASSERT_EQ(strings.size(), 3);
    GXF_ASSERT_STREQ(strings[0].c_str(), "Hello");
    GXF_ASSERT_STREQ(strings[1].c_str(), "world,");
    GXF_ASSERT_STREQ(strings[2].c_str(), "GXF");

    const auto& my_doubles = my_doubles_.get();
    GXF_ASSERT_EQ(my_doubles.size(), 3);
    GXF_ASSERT_EQ(my_doubles[0].size(), 3);
    GXF_ASSERT_EQ(my_doubles[1].size(), 0);
    GXF_ASSERT_EQ(my_doubles[2].size(), 1);
    GXF_ASSERT_EQ(my_doubles[0][0].d, 4.2);
    GXF_ASSERT_EQ(my_doubles[0][1].d, -5.2);
    GXF_ASSERT_EQ(my_doubles[0][2].d, 0.0);
    GXF_ASSERT_EQ(my_doubles[2][0].d, 100.0);

    const auto& segments = segments_.get();
    GXF_ASSERT_EQ(segments.size(), 2);
    GXF_ASSERT_EQ(segments[0].first[0], -100.0);
    GXF_ASSERT_EQ(segments[0].first[1], 0.0);
    GXF_ASSERT_EQ(segments[0].second[0], 100.0);
    GXF_ASSERT_EQ(segments[0].second[1], 0.0);
    GXF_ASSERT_EQ(segments[1].first[0], 0.0);
    GXF_ASSERT_EQ(segments[1].first[1], 0.0);
    GXF_ASSERT_EQ(segments[1].second[0], 20.0);
    GXF_ASSERT_EQ(segments[1].second[1], 2.0);

    return GXF_SUCCESS;
  }

  gxf_result_t registerInterface(Registrar* registrar) override {
    ParameterInfo<std::vector<double>> float64_vector_info {
      "vector_of_float",
      "testing float vector",
      "description for vector_of_float",
      "linux_x86_64, linux_aarch64",
    };
    float64_vector_info.rank = 1;
    float64_vector_info.shape[0] = 4;

    Expected<void> result;
    result &= registrar->parameter(integers_, "integers");
    result &= registrar->parameter(my_unsigned_int8_, "my_unsigned_int8");
    result &= registrar->parameter(strings_, "strings");
    result &= registrar->parameter(my_doubles_, "my_doubles");
    result &= registrar->parameter(segments_, "segments");
    result &= registrar->parameter(rank_2_vector_, "rank_2_vector");
    result &= registrar->parameter(rank_3_vector_, "rank_3_vector");
    result &= registrar->parameter(floats_, float64_vector_info);
    ParameterInfo<std::vector<Handle<Allocator>>> vector_info {
      "vector_of_handles",
      "headline for vector_of_handles",
      "description for vector_of_handles",
      "linux_x86_64, linux_aarch64",
    };
    vector_info.flags = GXF_PARAMETER_FLAGS_OPTIONAL;
    result &= registrar->parameter(vector_of_handles_, vector_info);
    result &= registrar->parameter(rank_2_array_, "rank_2_array");
    ParameterInfo<std::array<Handle<Allocator>, 2>> array_info {
      "array_of_two_handles",
      "headline for array_of_two_handles",
      "description for array_of_two_handles",
      "linux_x86_64, linux_aarch64",
    };
    array_info.flags = GXF_PARAMETER_FLAGS_OPTIONAL;
    result &= registrar->parameter(array_of_two_handles_, array_info);
    return ToResultCode(result);
  }

 private:
  Parameter<std::vector<int>> integers_;
  Parameter<std::vector<double>> floats_;
  Parameter<std::vector<uint8_t>> my_unsigned_int8_;
  Parameter<std::vector<std::string>> strings_;
  Parameter<std::vector<std::vector<MyDouble>>> my_doubles_;
  Parameter<std::vector<std::pair<std::array<double, 2>, std::array<double, 2>>>> segments_;
  Parameter<std::vector<std::vector<uint64_t>>> rank_2_vector_;
  Parameter<std::vector<std::vector<std::vector<uint64_t>>>> rank_3_vector_;
  Parameter<std::vector<Handle<Allocator>>> vector_of_handles_;
  Parameter<std::array<std::array<uint64_t, 2>, 1>> rank_2_array_;
  Parameter<std::array<Handle<Allocator>, 2>> array_of_two_handles_;
};

// Tests various features around handle parameters
class TestHandleParameter : public Component {
 public:
  gxf_result_t initialize() override {
    // Get the two pools
    const gxf_uid_t pool_cid = entity().get<Allocator>("pool").value().cid();
    const gxf_uid_t other_pool_cid = entity().get<Allocator>("other_pool").value().cid();
    GXF_ASSERT_NE(pool_cid, other_pool_cid);

    // Check that the handle parameter was correctly set
    GXF_ASSERT(pool_.try_get(), "pool not set");
    GXF_ASSERT_EQ(pool_.try_get()->cid(), pool_cid);
    GXF_ASSERT_EQ(pool_->cid(), pool_cid);

    // Check that C API gives the same result
    {
      gxf_uid_t handle_cid;
      GXF_ASSERT_SUCCESS(GxfParameterGetHandle(context(), cid(), "pool", &handle_cid));
      GXF_ASSERT_EQ(handle_cid, pool_cid);
    }

    // Change via C API
    GXF_ASSERT_SUCCESS(GxfParameterSetHandle(context(), cid(), "pool", other_pool_cid));

    // Check that C API gives the correct new result
    {
      gxf_uid_t handle_cid;
      GXF_ASSERT_SUCCESS(GxfParameterGetHandle(context(), cid(), "pool", &handle_cid));
      GXF_ASSERT_EQ(handle_cid, other_pool_cid);
    }

    // Check that C++ API gives the correct new result
    GXF_ASSERT(pool_.try_get(), "pool not set");
    GXF_ASSERT_EQ(pool_.try_get()->cid(), other_pool_cid);
    GXF_ASSERT_EQ(pool_->cid(), other_pool_cid);

    return GXF_SUCCESS;
  }

  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(pool_, "pool");
    return ToResultCode(result);
  }

  Parameter<Handle<Allocator>> pool_;
};

}  // namespace test
}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_TEST_EXTENSIONS_TEST_PARAMETERS_HPP_
