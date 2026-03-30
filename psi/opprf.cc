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

#include "psi/opprf.h"

#include "yacl/crypto/rand/rand.h"
#include "yacl/crypto/tools/prg.h"

#include "libspu/core/prelude.h"

#include "psi/rr22/okvs/paxos.h"

namespace psi::opprf {

OpprfConfig DefaultOpprfConfig() {
  OpprfConfig config;
  config.opprf_bin_size = 1 << 14;
  config.opprf_stats_security = 40;
  config.opprf_paxos_weigth = 3;

  config.num_threads = 8;
  config.is_send_fast_mode = true;
  return config;
}

OPPRFReceiver::OPPRFReceiver(const OpprfConfig& config, int64_t send_numel,
                             int64_t recv_numel,
                             std::shared_ptr<yacl::link::Context> conn)
    : config_(config), send_numel_(send_numel), recv_numel_(recv_numel) {
  SPU_ENFORCE(send_numel_ > 0);
  SPU_ENFORCE(recv_numel_ > 0);

  oprf_receiver_ = std::make_unique<psi::rr22::Rr22OprfReceiver>(
      config.opprf_bin_size, config.opprf_stats_security,
      config.is_send_fast_mode ? psi::rr22::Rr22PsiMode::FastMode
                               : psi::rr22::Rr22PsiMode::LowCommMode);
}

OPPRFSender::OPPRFSender(const OpprfConfig& config, int64_t send_numel,
                         int64_t recv_numel,
                         std::shared_ptr<yacl::link::Context> conn)
    : config_(config), send_numel_(send_numel), recv_numel_(recv_numel) {
  SPU_ENFORCE(send_numel_ > 0);
  SPU_ENFORCE(recv_numel_ > 0);

  oprf_sender_ = std::make_unique<psi::rr22::Rr22OprfSender>(
      config.opprf_bin_size, config.opprf_stats_security,
      config.is_send_fast_mode ? psi::rr22::Rr22PsiMode::FastMode
                               : psi::rr22::Rr22PsiMode::LowCommMode);
}

void OPPRFSender::SendBatch(absl::Span<const uint128_t> keys,
                            absl::Span<uint128_t> values, int64_t batch_size,
                            std::shared_ptr<yacl::link::Context> conn) {
  namespace okvs = psi::rr22::okvs;
  namespace yc = yacl::crypto;
  SPU_ENFORCE_EQ(keys.size() * batch_size, values.size());
  SPU_ENFORCE_EQ((int64_t)keys.size(), send_numel_);

  std::vector<uint128_t> _out;
  oprf_sender_->Init(conn, recv_numel_, config_.num_threads);
  {
    auto dummy = oprf_sender_->Send(conn, keys);
    _out = oprf_sender_->Eval(keys, absl::MakeSpan(dummy));
  }
  auto out = absl::MakeSpan(_out);

  // doubly obvlivious
  auto seed = yc::SecureRandSeed();
  conn->SendAsync(conn->NextRank(),
                  yacl::ByteContainerView(&seed, sizeof(uint128_t)),
                  "send:seed");
  okvs::Baxos baxos;
  baxos.Init(send_numel_, config_.opprf_bin_size, config_.opprf_paxos_weigth,
             config_.opprf_stats_security, okvs::PaxosParam::DenseType::GF128,
             seed);

  // double-randomness
  auto double_rnd = std::make_shared<yc::Prg<uint8_t>>(yc::SecureRandSeed());
  std::vector<uint128_t> okvs_payload(send_numel_);

  for (int64_t b = 0; b < batch_size; ++b) {
    std::vector<uint128_t> okvs_hint(baxos.size());

    // auto this_batch = values.subspan(b * send_numel_, send_numel_);
    for (int64_t i = 0; i < send_numel_; ++i) {
      // payload[i] = values[i] - OPRF(keys[i])
      okvs_payload[i] = values[b * send_numel_ + i] ^ out[i];
    }

    baxos.Solve(absl::MakeConstSpan(keys), absl::MakeSpan(okvs_payload),
                absl::MakeSpan(okvs_hint), double_rnd);

    conn->SendAsync(conn->NextRank(),
                    yacl::ByteContainerView(
                        okvs_hint.data(), sizeof(uint128_t) * okvs_hint.size()),
                    "send:hint");
  }
}

void OPPRFReceiver::RecvBatch(absl::Span<const uint128_t> keys,
                              int64_t batch_size,
                              std::shared_ptr<yacl::link::Context> conn,
                              absl::Span<uint128_t> out) {
  namespace okvs = psi::rr22::okvs;
  namespace yc = yacl::crypto;
  SPU_ENFORCE_EQ(keys.size() * batch_size, out.size());
  SPU_ENFORCE_EQ((int64_t)keys.size(), recv_numel_);

  oprf_receiver_->Init(conn, recv_numel_, config_.num_threads);
  auto oprf = oprf_receiver_->Recv(conn, keys);

  okvs::Baxos baxos;
  auto buffer = conn->Recv(conn->NextRank(), "recv:seed");
  SPU_ENFORCE_EQ((size_t)buffer.size(), sizeof(uint128_t));
  baxos.Init(send_numel_, config_.opprf_bin_size, config_.opprf_paxos_weigth,
             config_.opprf_stats_security, okvs::PaxosParam::DenseType::GF128,
             buffer.data<uint128_t>()[0]);

  for (int64_t b = 0; b < batch_size; ++b) {
    yacl::Buffer buffer =
        conn->Recv(conn->NextRank(), fmt::format("recv:hint"));

    SPU_ENFORCE_EQ((size_t)buffer.size(), baxos.size() * sizeof(uint128_t));

    auto okvs_hint = absl::MakeSpan(buffer.data<uint128_t>(),
                                    buffer.size() / sizeof(uint128_t));
    auto out_sub = out.subspan(b * recv_numel_, recv_numel_);
    baxos.Decode(keys, out_sub, okvs_hint);

    for (int64_t i = 0; i < recv_numel_; ++i) {
      out_sub[i] = oprf[i] ^ out_sub[i];
    }
  }
}

}  // namespace psi::opprf
