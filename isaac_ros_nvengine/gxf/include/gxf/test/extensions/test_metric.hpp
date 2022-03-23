/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_TEST_EXTENSIONS_TEST_METRIC_HPP
#define NVIDIA_GXF_TEST_EXTENSIONS_TEST_METRIC_HPP

#include "common/assert.hpp"
#include "gtest/gtest.h"
#include "gxf/std/codelet.hpp"
#include "gxf/std/metric.hpp"

namespace nvidia {
namespace gxf {
namespace test {

// Test metric logger codelet that computes and writes to a set of metrics on every tick.
class TestMetricLogger : public Codelet {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override {
    Expected<void> result;
    result &= registrar->parameter(sample_metric_parameter_, "sample_metric_parameter");
    result &= registrar->parameter(sample_metric_mean_, "sample_metric_mean");
    result &= registrar->parameter(sample_metric_rms_, "sample_metric_rms");
    result &= registrar->parameter(sample_metric_abs_max_, "sample_metric_abs_max");
    result &= registrar->parameter(sample_metric_max_, "sample_metric_max");
    result &= registrar->parameter(sample_metric_min_, "sample_metric_min");
    result &= registrar->parameter(sample_metric_sum_, "sample_metric_sum");
    result &= registrar->parameter(sample_metric_fixed_, "sample_metric_fixed");
    result &= registrar->parameter(sample_metric_fail_threshold_, "sample_metric_fail_threshold");
    return ToResultCode(result);
  }

  gxf_result_t start() override {
    tick_count_ = 0;
    sample_metric_mean_->setMeanAggregationFunction();
    sample_metric_rms_->setRootMeanSquareAggregationFunction();
    sample_metric_abs_max_->setAbsMaxAggregationFunction();
    sample_metric_max_->setMaxAggregationFunction();
    sample_metric_min_->setMinAggregationFunction();
    sample_metric_sum_->setSumAggregationFunction();
    sample_metric_fixed_->setFixedAggregationFunction();
    return GXF_SUCCESS;
  }

  gxf_result_t tick() override {
    sample_metric_parameter_->record(tick_count_);
    sample_metric_mean_->record(tick_count_);
    sample_metric_rms_->record(tick_count_);
    sample_metric_abs_max_->record(-1.0 * tick_count_);
    sample_metric_max_->record(tick_count_);
    sample_metric_min_->record(tick_count_);
    sample_metric_sum_->record(tick_count_);
    sample_metric_fixed_->record(tick_count_);
    sample_metric_fail_threshold_->record(tick_count_);
    tick_count_ += 1.0;
    return GXF_SUCCESS;
  }

  gxf_result_t stop() override {
    Expected<void> result;
    result &= verifyMetricResult(sample_metric_parameter_, 49.5, true);
    result &= verifyMetricResult(sample_metric_mean_, 49.5, true);
    result &= verifyMetricResult(sample_metric_rms_, 57.301832, true);
    result &= verifyMetricResult(sample_metric_abs_max_, 99.0, true);
    result &= verifyMetricResult(sample_metric_max_, 99.0, true);
    result &= verifyMetricResult(sample_metric_min_, 0.0, true);
    result &= verifyMetricResult(sample_metric_sum_, 4950.0, true);
    result &= verifyMetricResult(sample_metric_fixed_, 99.0, true);
    result &= verifyMetricResult(sample_metric_fail_threshold_, 4950.0, false);
    return ToResultCode(result);
  }

  Expected<void> verifyMetricResult(Handle<Metric> metric, double expected_aggregated_value,
                                    bool expected_success) {
    const auto maybe_aggregated_value = metric->getAggregatedValue();
    if (!maybe_aggregated_value) {
      gxf::ForwardError(maybe_aggregated_value);
    }
    // FIXME(dbhaskara): use GXF_ASSERT_NEAR when available
    EXPECT_NEAR(maybe_aggregated_value.value(), expected_aggregated_value, 1e-4);

    const auto maybe_success = metric->evaluateSuccess();
    if (!maybe_success) {
      gxf::ForwardError(maybe_success);
    }
    GXF_ASSERT_EQ(maybe_success.value(), expected_success);
    return Success;
  }

 private:
  Parameter<Handle<Metric>> sample_metric_parameter_;
  Parameter<Handle<Metric>> sample_metric_mean_;
  Parameter<Handle<Metric>> sample_metric_rms_;
  Parameter<Handle<Metric>> sample_metric_abs_max_;
  Parameter<Handle<Metric>> sample_metric_max_;
  Parameter<Handle<Metric>> sample_metric_min_;
  Parameter<Handle<Metric>> sample_metric_sum_;
  Parameter<Handle<Metric>> sample_metric_fixed_;
  Parameter<Handle<Metric>> sample_metric_fail_threshold_;

  // Tracks number of ticks so far
  double tick_count_;
};

}  // namespace test
}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_TEST_EXTENSIONS_TEST_METRIC_HPP
