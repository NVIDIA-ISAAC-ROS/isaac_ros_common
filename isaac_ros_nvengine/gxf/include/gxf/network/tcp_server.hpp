/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#ifndef NVIDIA_GXF_NETWORK_TCP_SERVER_HPP_
#define NVIDIA_GXF_NETWORK_TCP_SERVER_HPP_

#include <string>
#include <unordered_map>
#include <vector>

#include "gxf/network/tcp_server_socket.hpp"
#include "gxf/serialization/entity_serializer.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace gxf {

// Codelet that functions as a server in a TCP connection
class TcpServer : public Codelet {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override { return ToResultCode(server_socket_.close()); }

  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

 private:
  Parameter<std::vector<Handle<Receiver>>> receivers_;
  Parameter<std::vector<Handle<Transmitter>>> transmitters_;
  Parameter<std::vector<Handle<ComponentSerializer>>> serializers_;
  Parameter<std::string> address_;
  Parameter<int> port_;
  Parameter<uint64_t> timeout_ms_;
  Parameter<uint64_t> maximum_attempts_;

  // Maps channel IDs to transmitters
  std::unordered_map<uint64_t, Handle<Transmitter>> channel_map_;
  // Entity serializer
  EntitySerializer entity_serializer_;
  // TCP server socket
  TcpServerSocket server_socket_;
  // TCP client socket
  TcpClientSocket client_socket_;
  // Execution timestamp for measuring connection timeout
  int64_t timestamp_;
};

}  // namespace gxf
}  // namespace nvidia

#endif  // NVIDIA_GXF_NETWORK_TCP_SERVER_HPP_
