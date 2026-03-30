// Copyright 2023 TikTok Pte. Ltd.
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

#pragma once

#include <memory>

#include "psi/rr22/rr22_oprf.h"

namespace psi::opprf {

struct OpprfConfig {
  size_t opprf_bin_size;
  size_t opprf_stats_security;
  size_t opprf_paxos_weigth;

  size_t num_threads;
  bool is_send_fast_mode;
};

OpprfConfig DefaultOpprfConfig();

class OPPRFReceiver {
 public:
  OPPRFReceiver(const OpprfConfig& config, int64_t send_numel,
                int64_t recv_numel, std::shared_ptr<yacl::link::Context> conn);

  void Recv(absl::Span<const uint128_t> keys,
            std::shared_ptr<yacl::link::Context> conn,
            absl::Span<uint128_t> out) {
    RecvBatch(keys, 1, conn, out);
  }

  void RecvBatch(absl::Span<const uint128_t> keys, int64_t batch_size,
                 std::shared_ptr<yacl::link::Context> conn,
                 absl::Span<uint128_t> out);

 private:
  OpprfConfig config_;
  int64_t send_numel_;
  int64_t recv_numel_;
  std::unique_ptr<psi::rr22::Rr22OprfReceiver> oprf_receiver_;
};

class OPPRFSender {
 public:
  OPPRFSender(const OpprfConfig& config, int64_t send_numel, int64_t recv_numel,
              std::shared_ptr<yacl::link::Context> conn);

  void Send(absl::Span<const uint128_t> keys, absl::Span<uint128_t> values,
            std::shared_ptr<yacl::link::Context> conn) {
    SendBatch(keys, values, 1, conn);
  }

  // values is reshaped to two-dim matrix values[B][N]
  // NOTE: the values will be overwrited
  void SendBatch(absl::Span<const uint128_t> keys, absl::Span<uint128_t> values,
                 int64_t batch_size, std::shared_ptr<yacl::link::Context> conn);

 private:
  OpprfConfig config_;
  int64_t send_numel_;
  int64_t recv_numel_;
  std::unique_ptr<psi::rr22::Rr22OprfSender> oprf_sender_;
};

}  // namespace psi::opprf
