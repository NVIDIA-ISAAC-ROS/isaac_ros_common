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
#ifndef NVIDIA_COMMON_LOGGER_HPP_
#define NVIDIA_COMMON_LOGGER_HPP_

#include <cstdarg>
#include <cstdio>
#include <vector>

// Logs a debug message
#define GXF_LOG_DEBUG(...) \
  ::nvidia::Log(__FILE__, __LINE__, ::nvidia::Severity::DEBUG, __VA_ARGS__)

// Logs an informational message
#define GXF_LOG_INFO(...) \
  ::nvidia::Log(__FILE__, __LINE__, ::nvidia::Severity::INFO, __VA_ARGS__)

// Logs a warning
#define GXF_LOG_WARNING(...) \
  ::nvidia::Log(__FILE__, __LINE__, ::nvidia::Severity::WARNING, __VA_ARGS__)

// Logs an error
#define GXF_LOG_ERROR(...) \
  ::nvidia::Log(__FILE__, __LINE__, ::nvidia::Severity::ERROR, __VA_ARGS__)

namespace nvidia {

// Indicates the level of severity for a log message
enum class Severity {
  // A utility case which can be used for 'SetSeverity' to disable all severity levels.
  NONE = -2,
  // A utility case which can be used for 'Redirect' and 'SetSeverity'
  ALL = -1,
  // The five different log severities in use from most severe to least severe.
  PANIC = 0,  // Need to start at 0
  ERROR,
  WARNING,
  INFO,
  DEBUG,
  // A utility case representing the number of log levels used internally.
  COUNT
};

// Function which is used for logging. It can be changed to intercept the logged messages.
extern void (*LoggingFunction)(const char* file, int line, Severity severity, const char* log);

// Redirects the output for a given log severity.
void Redirect(std::FILE* file, Severity severity = Severity::ALL);

// Sets global log severity thus effectively disabling all logging with lower severity
void SetSeverity(Severity severity);

// Converts the message and argument into a string and pass it to LoggingFunction.
template<typename... Args>
void Log(const char* file, int line, Severity severity, const char* txt, ...) {
  va_list args1;
  va_start(args1, txt);
  va_list args2;
  va_copy(args2, args1);
  std::vector<char> buf(1 + std::vsnprintf(NULL, 0, txt, args1));
  va_end(args1);
  std::vsnprintf(buf.data(), buf.size(), txt, args2);
  va_end(args2);
  LoggingFunction(file, line, severity, buf.data());
}

}  // namespace nvidia

#endif  // NVIDIA_COMMON_LOGGER_HPP_
