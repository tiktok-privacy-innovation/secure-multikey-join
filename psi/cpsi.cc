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
//
#include "psi/cpsi.h"

#include <algorithm>
#include <cstdint>
#include <future>
#include <type_traits>

#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"

#include "psi/cuckoo_index.h"
#include "psi/eqt.h"
#include "psi/permute.h"
namespace psi {

constexpr spu::FieldType kRandTagType = spu::FM128;

template <typename T>
inline T CeilDiv(T a, T b) {
  YACL_ENFORCE(b > 0);
  return (a + b - 1) / b;
}

uint128_t PackU64(uint64_t x, uint64_t y) {
  return (static_cast<uint128_t>(x) << 64) | y;
}

std::array<uint64_t, 2> UnpackU64(uint128_t u) {
  return {static_cast<uint64_t>(u >> 64), static_cast<uint64_t>(u)};
}

uint128_t HashToU128(const std::array<uint128_t, 2>& x) {
  return yacl::crypto::Blake3_128({&x[0], sizeof(x)});
}

CuckooIndex::Options DefaultCuckooIndexOptions(int64_t numel) {
  constexpr size_t kNumStash = 0;
  constexpr size_t kNumSimpleHash = 3;
  constexpr float kExpandFactor = 1.27;
  return CuckooIndex::Options{(size_t)numel, kNumStash, kNumSimpleHash,
                              kExpandFactor};
}

CircuitPSIBase::CircuitPSIBase(std::array<OTProt, 2> ot_pair)
    : ot_(ot_pair[0]), duplx_ot_(ot_pair[1]) {
  rank_ = ot_->GetCommunicator()->lctx()->Rank();
}

CircuitPSIServer::CircuitPSIServer(std::array<OTProt, 2> ot_pair)
    : CircuitPSIBase(ot_pair) {}

CircuitPSIClient::CircuitPSIClient(std::array<OTProt, 2> ot_pair)
    : CircuitPSIBase(ot_pair) {}

void CircuitPSIServer::PermuteInplace(absl::Span<uint8_t> eq_bits) const {
  mpc::PermuteBase::Meta meta;
  meta.numel = eq_bits.size();
  meta.is_arithmetic = false;
  meta.payload_width = 1;

  mpc::PermuteSender sender(meta);
  std::vector<uint8_t> perm_eq_bits(meta.numel);
  sender.Send(eq_bits, absl::MakeSpan(perm_eq_bits), ot_);
  std::copy_n(perm_eq_bits.data(), perm_eq_bits.size(), eq_bits.data());
}

bool IsValidPermutation(absl::Span<const int32_t> perm) {
  int32_t n = perm.size();
  if (!std::all_of(perm.begin(), perm.end(),
                   [n](int32_t x) { return x >= 0 && x < n; })) {
    return false;
  }

  std::vector<int32_t> copy(perm.begin(), perm.end());
  std::sort(copy.begin(), copy.end());
  for (int32_t i = 1; i < n; ++i) {
    if (copy[i] == copy[i - 1]) {
      return false;
    }
  }
  return true;
}

void CircuitPSIClient::PermuteInplace(absl::Span<uint8_t> eq_bits,
                                      absl::Span<const int32_t> perm) const {
  YACL_ENFORCE(IsValidPermutation(perm));

  size_t sent = ot_->GetCommunicator()->lctx()->GetStats()->sent_bytes;
  size_t recv = ot_->GetCommunicator()->lctx()->GetStats()->recv_bytes;

  mpc::PermuteBase::Meta meta;
  meta.numel = eq_bits.size();
  meta.is_arithmetic = false;
  meta.payload_width = 1;

  mpc::PermuteReceiver receiver(meta, perm);
  std::vector<uint8_t> perm_eq_bits(meta.numel);
  receiver.Recv(eq_bits, absl::MakeSpan(perm_eq_bits), ot_);
  std::copy_n(perm_eq_bits.data(), perm_eq_bits.size(), eq_bits.data());

  sent = ot_->GetCommunicator()->lctx()->GetStats()->sent_bytes - sent;
  recv = ot_->GetCommunicator()->lctx()->GetStats()->recv_bytes - recv;
  SPDLOG_INFO("Perm 1-bit n={}, {} MiB", meta.numel,
              (sent + recv) / 1024. / 1024.);
}

spu::NdArrayRef CircuitPSIServer::PermuteTranspose(
    const spu::NdArrayRef& payloads, int64_t keep_front_size) const {
  size_t sent = ot_->GetCommunicator()->lctx()->GetStats()->sent_bytes;
  size_t recv = ot_->GetCommunicator()->lctx()->GetStats()->recv_bytes;

  int64_t num_payloads = payloads.shape()[0];
  int64_t num_elt = payloads.shape()[1];

  mpc::PermuteBase::Meta meta;
  meta.numel = num_elt;
  meta.is_arithmetic = true;
  meta.payload_width = 64;

  if (keep_front_size < 0) {
    keep_front_size = num_elt;
  }
  YACL_ENFORCE(keep_front_size <= num_elt);

  mpc::PermuteSender sender(meta);

  spu::NdArrayRef perm_payloads(payloads.eltype(),
                                {keep_front_size, num_payloads});

  std::vector<uint64_t> perm_output(meta.numel);
  for (int64_t b = 0; b < num_payloads; ++b) {
    auto perm_input = payloads.slice({b, 0}, {b + 1, num_elt}, {1, 1});

    sender.Send(absl::MakeConstSpan(perm_input.data<uint64_t>(), num_elt),
                absl::MakeSpan(perm_output), ot_);

    // transpose here
    for (int64_t i = 0; i < keep_front_size; ++i) {
      perm_payloads.at<uint64_t>({i, b}) = perm_output[i];
    }
  }

  sent = ot_->GetCommunicator()->lctx()->GetStats()->sent_bytes - sent;
  recv = ot_->GetCommunicator()->lctx()->GetStats()->recv_bytes - recv;

  SPDLOG_INFO("Perm Arith n={}, ell={}, {} MiB", meta.numel,
              meta.payload_width * num_payloads, (sent + recv) / 1024. / 1024.);
  return perm_payloads;
}

spu::NdArrayRef CircuitPSIClient::PermuteTranspose(
    const spu::NdArrayRef& payloads, int64_t keep_front_size,
    absl::Span<const int32_t> perm) const {
  int64_t num_payloads = payloads.shape()[0];
  int64_t num_elt = payloads.shape()[1];

  mpc::PermuteBase::Meta meta;
  meta.numel = num_elt;
  meta.is_arithmetic = true;
  meta.payload_width = 64;

  if (keep_front_size < 0) {
    keep_front_size = num_elt;
  }
  YACL_ENFORCE(keep_front_size <= num_elt);

  YACL_ENFORCE_EQ(payloads.numel(),
                  static_cast<int64_t>(num_payloads * perm.size()),
                  "payloads.size != perm.size");

  spu::NdArrayRef perm_payloads(payloads.eltype(),
                                {keep_front_size, num_payloads});
  mpc::PermuteReceiver receiver(meta, absl::MakeSpan(perm));

  std::vector<uint64_t> perm_output(meta.numel);
  for (int64_t b = 0; b < num_payloads; ++b) {
    auto perm_input = payloads.slice({b, 0}, {b + 1, num_elt}, {1, 1});

    receiver.Recv(absl::MakeConstSpan(perm_input.data<uint64_t>(), num_elt),
                  absl::MakeSpan(perm_output), ot_);

    // transpose here
    for (int64_t i = 0; i < keep_front_size; ++i) {
      perm_payloads.at<uint64_t>({i, b}) = perm_output[i];
    }
  }

  return perm_payloads;
}

std::tuple<std::vector<uint8_t>, spu::NdArrayRef> CircuitPSIServer::Send(
    absl::Span<const uint128_t> keys, const spu::NdArrayRef& payload,
    int64_t client_numel,
    const std::shared_ptr<yacl::link::Context>& conn) const {
  YACL_ENFORCE_EQ((int64_t)keys.size(), payload.shape()[0],
                  "#keys = {} mismatch payload shape {}", keys.size(),
                  payload.shape());
  YACL_ENFORCE_EQ(payload.elsize(), 8UL, "only 64b payload is supported");
  YACL_ENFORCE_EQ(payload.ndim(), 2UL, "only 2-dim payload is supported");

  const int64_t num_bins = [&]() {
    auto buffer = conn->Recv(conn->NextRank(), "recv:num_bins");
    YACL_ENFORCE_EQ((size_t)buffer.size(), sizeof(int64_t));
    return buffer.data<int64_t>()[0];
  }();

  const int64_t srv_numel = keys.size();
  const int64_t num_payloads = payload.shape()[1];

  // Simple Hash
  const int64_t num_simple_hash = DefaultCuckooIndexOptions(0).num_hash;
  const int64_t sh_tbl_sze = srv_numel * num_simple_hash;
  using SHTabRow = std::vector<std::pair<int64_t, int64_t>>;
  std::vector<SHTabRow> simple_hash_tbl(num_bins);

  for (int64_t i = 0; i < srv_numel; ++i) {
    psi::CuckooIndex::HashRoom room(keys[i]);
    for (uint8_t h_idx = 0; h_idx < num_simple_hash; ++h_idx) {
      auto bin_idx = room.GetHash(h_idx) % num_bins;
      simple_hash_tbl[bin_idx].emplace_back(i, (int64_t)h_idx);
    }
  }

  // random tags for each bin
  spu::NdArrayRef bin_tags = spu::mpc::ring_rand(kRandTagType, {num_bins});
  // arithmetic masks for each bin and each payload columns
  using Type = uint64_t;  // For now u64 payload only
  auto u64_share_type = spu::makeType<spu::mpc::cheetah::AShrTy>(spu::FM64);
  auto u128_share_type = spu::makeType<spu::mpc::cheetah::AShrTy>(spu::FM128);
  const int64_t opprf_batch =
      1 + CeilDiv<int64_t>(num_payloads, 2);  // count GF(128)

  // Prepare the OPPRF key-value pairs
  spu::NdArrayRef bin_mask =
      spu::mpc::ring_rand(spu::FM64, {num_payloads, num_bins});
  spu::NdArrayRef opprf_keys(u128_share_type, {sh_tbl_sze});
  spu::NdArrayRef opprf_values(u128_share_type, {opprf_batch, sh_tbl_sze});

  int64_t entry_id = 0;
  for (int64_t bin_id = 0; bin_id < num_bins; ++bin_id) {
    // bin_index to (sample_index, hash_index)
    for (auto [smp_id, hash_id] : simple_hash_tbl[bin_id]) {
      std::array<uint128_t, 2> key_buf;
      key_buf[0] = hash_id;
      key_buf[1] = keys[smp_id];
      // OPPRF key = Hash(hash_index || Hash(sample))
      opprf_keys.at<uint128_t>(entry_id) = HashToU128(key_buf);

      for (int64_t batch = 0; batch + 1 < opprf_batch; ++batch) {
        int64_t s0 = payload.at<Type>({smp_id, batch << 1}) -
                     bin_mask.at<Type>({batch << 1, bin_id});
        int64_t s1 = 0;

        if (2 * batch + 1 < num_payloads) {
          s1 = payload.at<Type>({smp_id, (batch << 1) + 1}) -
               bin_mask.at<Type>({(batch << 1) + 1, bin_id});
        }

        opprf_values.at<uint128_t>({batch, entry_id}) = PackU64(s0, s1);
      }

      // The last batch is the random tag
      opprf_values.at<uint128_t>({opprf_batch - 1, entry_id}) =
          bin_tags.at<uint128_t>(bin_id);
      entry_id += 1;
    }
  }

  auto opprf_cfg = opprf::DefaultOpprfConfig();
  opprf::OPPRFSender opprf_sender(opprf_cfg, sh_tbl_sze, client_numel, conn);

  size_t sent = conn->GetStats()->sent_bytes;
  size_t recv = conn->GetStats()->recv_bytes;
  opprf_sender.SendBatch(
      absl::MakeSpan(opprf_keys.data<uint128_t>(), opprf_keys.numel()),
      absl::MakeSpan(opprf_values.data<uint128_t>(), opprf_values.numel()),
      opprf_batch, conn);
  sent = conn->GetStats()->sent_bytes - sent;
  recv = conn->GetStats()->recv_bytes - recv;
  SPDLOG_INFO("OPPRF batch={} exchange {} MiB", opprf_batch,
              (sent + recv) / 1024. / 1024.);

  const int64_t eq_bw = std::min<int64_t>(
      128, opprf_cfg.opprf_stats_security + absl::bit_width((size_t)srv_numel));

  auto bin_eqt = EqualTest(bin_tags, eq_bw);

  PermuteInplace(absl::MakeSpan(bin_eqt));
  bin_mask = PermuteTranspose(bin_mask, client_numel);

  std::vector<uint8_t> sample_eqt(client_numel);
  std::copy_n(bin_eqt.data(), client_numel, sample_eqt.data());

  return {sample_eqt, bin_mask};
}

std::tuple<std::vector<uint8_t>, spu::NdArrayRef> CircuitPSIClient::Recv(
    absl::Span<const uint128_t> keys, const spu::Shape& srv_payload_shape,
    const std::shared_ptr<yacl::link::Context>& conn) const {
  YACL_ENFORCE_EQ(srv_payload_shape.ndim(), 2L,
                  "only 2-dim payload is supported");
  const int64_t srv_numel = srv_payload_shape[0];
  const int64_t num_payloads = srv_payload_shape[1];
  const int64_t opprf_batch =
      1 + CeilDiv<int64_t>(num_payloads, 2);  // count GF(128)

  int64_t client_numel = keys.size();
  YACL_ENFORCE_GE(srv_numel, client_numel,
                  "server #keys={} should larger than client #keys={}",
                  srv_numel, client_numel);

  CuckooIndex cck_index(DefaultCuckooIndexOptions(client_numel));
  cck_index.Insert(keys);
  CuckooIndexHelper cck_helper(cck_index, client_numel);

  int64_t num_bins = cck_helper.num_bin();

  conn->Send(conn->NextRank(), {&num_bins, sizeof(int64_t)}, "send:num_bins");

  auto tag_type = spu::makeType<spu::mpc::cheetah::AShrTy>(kRandTagType);
  auto u64_share_type = spu::makeType<spu::mpc::cheetah::AShrTy>(spu::FM64);
  auto u128_share_type = spu::makeType<spu::mpc::cheetah::AShrTy>(spu::FM128);

  // Simple Hash
  const int64_t num_simple_hash = DefaultCuckooIndexOptions(0).num_hash;
  const int64_t sh_tbl_sze = srv_numel * num_simple_hash;

  const auto& sample_to_bin_map = cck_helper.sample_to_bin_map();
  spu::NdArrayRef opprf_query(u128_share_type, {client_numel});
  for (int64_t i = 0; i < client_numel; ++i) {
    std::array<uint128_t, 2> key_buf;
    key_buf[0] = cck_index.bins()[sample_to_bin_map[i]].HashIdx();
    key_buf[1] = keys[i];
    opprf_query.at<uint128_t>(i) = HashToU128(key_buf);
  }

  auto opprf_cfg = opprf::DefaultOpprfConfig();
  opprf::OPPRFReceiver receiver(opprf_cfg, sh_tbl_sze, client_numel, conn);

  spu::NdArrayRef opprf_recv(u128_share_type, {opprf_batch, client_numel});
  receiver.RecvBatch(
      absl::MakeSpan(opprf_query.data<uint128_t>(), opprf_query.numel()),
      opprf_batch, conn,
      absl::MakeSpan(opprf_recv.data<uint128_t>(), opprf_recv.numel()));

  spu::NdArrayRef bin_tags(tag_type, {num_bins});
  spu::NdArrayRef bin_mask(u64_share_type, {num_payloads, num_bins});

  for (int64_t entry_id = 0; entry_id < client_numel; ++entry_id) {
    auto bin_id = sample_to_bin_map[entry_id];

    for (int64_t batch = 0; batch + 1 < opprf_batch; ++batch) {
      auto unpack = UnpackU64(opprf_recv.at<uint128_t>({batch, entry_id}));
      bin_mask.at<uint64_t>({2 * batch, bin_id}) = unpack[0];
      if (2 * batch + 1 < num_payloads) {
        bin_mask.at<uint64_t>({(2 * batch) + 1, bin_id}) = unpack[1];
      }
    }
    // last batch is the random tag
    bin_tags.at<uint128_t>(bin_id) =
        opprf_recv.at<uint128_t>({opprf_batch - 1, entry_id});
  }

  const int64_t eq_bw = std::min<int64_t>(
      128, opprf_cfg.opprf_stats_security + absl::bit_width((size_t)srv_numel));
  auto bin_eqt = EqualTest(bin_tags, eq_bw);

  const auto& perm = cck_helper.bin_to_sample_perm();
  PermuteInplace(absl::MakeSpan(bin_eqt), perm);
  bin_mask = PermuteTranspose(bin_mask, client_numel, perm);

  std::vector<uint8_t> sample_eqt(client_numel);
  std::copy_n(bin_eqt.data(), client_numel, sample_eqt.data());

  return {sample_eqt, bin_mask};
}

spu::NdArrayRef CircuitPSIBase::WaterFallUpdateInplace(
    absl::Span<uint8_t> already_match_indicator,
    absl::Span<const uint8_t> this_key_match_indicator,
    const spu::NdArrayRef& this_key_match_payload) const {
  size_t sent = ot_->GetCommunicator()->lctx()->GetStats()->sent_bytes;
  size_t recv = ot_->GetCommunicator()->lctx()->GetStats()->recv_bytes;

  YACL_ENFORCE_EQ(already_match_indicator.size(),
                  this_key_match_indicator.size());
  YACL_ENFORCE_EQ(this_key_match_payload.ndim(), 2UL,
                  "only 2-dim payload is supported");
  int64_t num_samples = this_key_match_payload.shape()[0];
  YACL_ENFORCE_EQ((int64_t)already_match_indicator.size(), num_samples);

  // (1 - already_match_indicator) ^ this_key_match_indicator
  // (1 - already_match_indicator) ^ (1 - this_key_match_indicator)
  auto btype = spu::makeType<spu::mpc::cheetah::BShrTy>(spu::FM32, 1);
  spu::NdArrayRef op0(btype, {num_samples});
  spu::NdArrayRef op1(btype, {num_samples});
  spu::NdArrayRef op2(btype, {num_samples});

  const uint8_t ONE = rank_ == 0 ? 1 : 0;

  spu::pforeach(0, num_samples, [&](int64_t i) {
    op0.data<uint32_t>()[i] = ONE ^ (already_match_indicator[i] & 1);
    op1.data<uint32_t>()[i] = this_key_match_indicator[i] & 1;
    op2.data<uint32_t>()[i] = ONE ^ (this_key_match_indicator[i] & 1);
  });

  // cAND0 = op0 ^ op1
  // cAND1 = op0 ^ op2
  auto cANDs = ot_->CorrelatedBitwiseAnd(op0, op1, op2);

  int64_t num_payloads = this_key_match_payload.shape()[1];

  std::vector<spu::NdArrayRef> updated_payload(num_payloads);

  for (int64_t b = 0; b < num_payloads; ++b) {
    auto payload_slice =
        this_key_match_payload.slice({0, b}, {num_samples, b + 1}, {1, 1})
            .reshape({num_samples});
    updated_payload[b] =
        ot_->Multiplexer(payload_slice, cANDs[0]).reshape({num_samples, 1});
  }

  spu::pforeach(0, num_samples, [&](int64_t i) {
    already_match_indicator[i] = ONE ^ (cANDs[1].data<uint32_t>()[i] & 1);
  });

  sent = ot_->GetCommunicator()->lctx()->GetStats()->sent_bytes - sent;
  recv = ot_->GetCommunicator()->lctx()->GetStats()->recv_bytes - recv;

  if (rank_ == 0) {
    SPDLOG_INFO("Waterfall Update n={}, ell={}, {} MiB", num_samples,
                this_key_match_payload.elsize() * 8 * num_payloads,
                (sent + recv) / 1024. / 1024.);
  }

  return updated_payload[0].concatenate(
      absl::MakeSpan(&updated_payload[1], updated_payload.size() - 1), 1);
}

std::vector<uint8_t> CircuitPSIBase::EqualTest(spu::NdArrayRef x,
                                               int64_t bw) const {
  size_t sent = ot_->GetCommunicator()->lctx()->GetStats()->sent_bytes;
  size_t recv = ot_->GetCommunicator()->lctx()->GetStats()->recv_bytes;
  const int64_t radix = bw > 64 ? 4 : 3;
  const int64_t numel = x.numel();
  // no splitting when the |input| < 2048
  const int64_t half_n =
      numel < 2048 ? numel : std::max<int64_t>(1, numel >> 1);

  // Duplx execution
  auto subtask = std::async([&]() {
    if (half_n == numel) {
      return spu::NdArrayRef();
    }

    psi::mpc::EqTProtocol eqt(duplx_ot_, radix);
    eqt.set_as_sender(rank_ == 0);
    auto out = eqt.Compute(x.slice({half_n}, {numel}, {1}), bw);
    return out;
  });

  mpc::EqTProtocol eqt(ot_, radix);
  eqt.set_as_sender(rank_ != 0);

  auto out0 = eqt.Compute(x.slice({0}, {half_n}, {1}), bw);
  auto out1 = subtask.get();

  std::vector<uint8_t> eq_out(numel);
  for (int64_t i = 0, j = 0; i < half_n; ++i, j += out0.elsize()) {
    eq_out[i] = out0.data<uint8_t>()[j];
  }
  for (int64_t i = half_n, j = 0; i < numel; ++i, j += out0.elsize()) {
    eq_out[i] = out1.data<uint8_t>()[j];
  }

  sent = ot_->GetCommunicator()->lctx()->GetStats()->sent_bytes - sent;
  recv = ot_->GetCommunicator()->lctx()->GetStats()->recv_bytes - recv;
  if (rank_ == 0) {
    SPDLOG_INFO("EqT n={}, ell={}, {} MiB", numel, bw,
                (sent + recv) / 1024. / 1024.);
  }
  return eq_out;
}

}  // namespace psi
