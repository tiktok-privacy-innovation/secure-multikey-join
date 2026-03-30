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

#include "psi/cpsi.h"

#include <algorithm>
#include <condition_variable>

#include "cpsi.h"
#include "gtest/gtest.h"
#include "yacl/crypto/rand/rand.h"
#include "yacl/utils/elapsed_timer.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/cheetah/nonlinear/equal_prot.h"
#include "libspu/mpc/cheetah/nonlinear/osn_prot.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/permute.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace psi::test {

class CPSITest
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, int64_t>> {
 public:
  void SetUp() override {
    int64_t svr_numel = std::get<0>(GetParam());
    int64_t clt_numel = std::get<1>(GetParam());
    int64_t num_payloads = std::get<2>(GetParam());

    if (svr_numel < clt_numel) {
      return;
    }

    yacl::crypto::Prg<uint128_t> prng(12345);

    svr_keys_.resize(svr_numel);
    clt_keys_.resize(clt_numel);

    prng.Fill(absl::MakeSpan(svr_keys_));
    prng.Fill(absl::MakeSpan(clt_keys_));

    {
      std::unordered_set<int64_t> matched;

      for (int64_t i = 0; i < clt_numel; i++) {
        // 12.5% match ratio
        if (prng() % 1024 >= 872) {
          int64_t j = prng() % svr_numel;
          while (matched.count(j)) {
            j = prng() % svr_numel;
          }
          matched.insert(j);

          matched_pairs_.emplace_back(i, j);
          clt_keys_[i] = svr_keys_[j];
        }
      }
    }

    svr_2nd_keys_.resize(svr_numel);
    clt_2nd_keys_.resize(clt_numel);
    prng.Fill(absl::MakeSpan(svr_2nd_keys_));
    prng.Fill(absl::MakeSpan(clt_2nd_keys_));

    {
      std::unordered_set<int64_t> matched;
      for (int64_t i = 0; i < clt_numel; i++) {
        // 12.5% match ratio
        if (prng() % 1024 >= 872) {
          int64_t j = prng() % svr_numel;
          while (matched.count(j)) {
            j = prng() % svr_numel;
          }
          matched.insert(j);

          matched_2nd_pairs_.emplace_back(i, j);
          clt_2nd_keys_[i] = svr_2nd_keys_[j];
        }
      }
    }

    svr_payloads_ = spu::mpc::ring_ones(spu::FM64, {svr_numel, num_payloads});
    for (int b = 0; b < num_payloads; ++b) {
      svr_payloads_.at<uint64_t>({2, b}) = 100;
    }
  }

  std::vector<uint8_t> ret_eq_bit_[2];
  spu::NdArrayRef ret_payload_[2];

  void ApplyOneKey(absl::Span<const uint128_t> clt_keys,
                   absl::Span<const uint128_t> svr_keys,
                   const spu::NdArrayRef& svr_payloads,
                   absl::Span<uint8_t> already_match_indicator0,
                   absl::Span<uint8_t> already_match_indicator1) {
    int64_t svr_numel = svr_keys.size();
    int64_t clt_numel = clt_keys.size();
    int64_t num_payloads = svr_payloads.shape()[1];

    std::vector<uint8_t> eq_bit[2];
    spu::NdArrayRef payload[2];

    spu::mpc::utils::simulate(
        2, [&](std::shared_ptr<yacl::link::Context> conn) {
          size_t sent = conn->GetStats()->sent_bytes;
          size_t recv = conn->GetStats()->recv_bytes;
          auto ot = std::make_shared<spu::mpc::cheetah::BasicOTProtocols>(
              conn, spu::CheetahOtKind::YACL_Ferret);

          std::shared_ptr<yacl::link::Context> conn2 = conn->Spawn();
          auto dup_ot = std::make_shared<spu::mpc::cheetah::BasicOTProtocols>(
              conn2, spu::CheetahOtKind::YACL_Ferret);

          int rank = conn->Rank();

          if (rank == 0) {
            CircuitPSIServer server({ot, dup_ot});
            auto recv = server.Send(svr_keys, svr_payloads, clt_numel, conn);
            eq_bit[rank] = std::get<0>(recv);
            payload[rank] = std::get<1>(recv);

            payload[rank] = server.WaterFallUpdateInplace(
                already_match_indicator0, absl::MakeSpan(eq_bit[rank]),
                payload[rank]);
          } else {
            CircuitPSIClient client({ot, dup_ot});
            auto recv = client.Recv(absl::MakeConstSpan(clt_keys),
                                    svr_payloads.shape(), conn);
            eq_bit[rank] = std::get<0>(recv);
            payload[rank] = std::get<1>(recv);

            payload[rank] = client.WaterFallUpdateInplace(
                already_match_indicator1, absl::MakeSpan(eq_bit[rank]),
                payload[rank]);
          }

          sent = conn->GetStats()->sent_bytes - sent;
          recv = conn->GetStats()->recv_bytes - recv;
          if (rank == 0) {
            SPDLOG_INFO("SingleKey CPSI: Svr={}, Client={}, ell={}, {} MiB",
                        svr_numel, clt_numel, num_payloads * 64,
                        (sent + recv) / 1024. / 1024.);
          }
        });

    ASSERT_EQ(eq_bit[0].size(), eq_bit[1].size());
    // ASSERT_EQ(eq_bit[0].size(), (size_t)clt_numel);
    ASSERT_EQ(payload[0].shape(), payload[1].shape());
    ASSERT_EQ(payload[0].ndim(), 2UL);
    // ASSERT_EQ(payload[0].shape()[0], clt_numel);
    ASSERT_EQ(payload[0].shape()[1], num_payloads);

    auto got_payload = spu::mpc::ring_add(payload[0], payload[1]);

    int64_t n = eq_bit[0].size();
    for (int64_t i = 0; i < n; ++i) {
      ASSERT_LE(eq_bit[0][i], 1);
      ASSERT_LE(eq_bit[1][i], 1);

      if (0 == (eq_bit[0][i] ^ eq_bit[1][i])) {
        for (int64_t b = 0; b < num_payloads; ++b) {
          uint64_t got0 = payload[0].at<uint64_t>({i, b});
          uint64_t got1 = payload[1].at<uint64_t>({i, b});
          ASSERT_NE(got0, got1);
          ASSERT_EQ(got0 + got1, 0);
        }
      }
    }

    ret_payload_[0] = payload[0];
    ret_payload_[1] = payload[1];
    ret_eq_bit_[0] = eq_bit[0];
    ret_eq_bit_[1] = eq_bit[1];
  }

  std::vector<uint128_t> svr_keys_;
  spu::NdArrayRef svr_payloads_;
  std::vector<uint128_t> clt_keys_;
  std::vector<std::pair<uint32_t, uint32_t>> matched_pairs_;

  std::vector<uint128_t> svr_2nd_keys_;
  std::vector<uint128_t> clt_2nd_keys_;
  std::vector<std::pair<uint32_t, uint32_t>> matched_2nd_pairs_;
};

INSTANTIATE_TEST_SUITE_P(
    PSI, CPSITest,
    testing::Combine(testing::Values(1 << 11, 1L << 20, 11774383),
                     testing::Values(1 << 10, 1 << 20, 11472479),
                     testing::Values(1, 2, 3, 4, 5, 7)),
    [](const testing::TestParamInfo<CPSITest::ParamType>& p) {
      return fmt::format("Server{}Client{}Payload{}", std::get<0>(p.param),
                         std::get<1>(p.param), std::get<2>(p.param));
    });

spu::NdArrayRef ViewFM64ArrayAsFM128(spu::NdArrayRef u64) {
  YACL_ENFORCE_EQ(u64.ndim(), 2UL);
  YACL_ENFORCE_EQ(u64.shape()[1], 2L);

  return spu::NdArrayRef(u64.buf(), spu::makeType<spu::RingTy>(spu::FM128),
                         spu::Shape{u64.shape()[0]},
                         spu::Strides{u64.strides()[0] / 2}, u64.offset());
}

TEST_P(CPSITest, SingleKey) {
  int64_t svr_numel = std::get<0>(GetParam());
  int64_t clt_numel = std::get<1>(GetParam());

  if (svr_numel < clt_numel) {
    return;
  }

  std::vector<uint8_t> already_match_indicator[2];
  already_match_indicator[0].resize(clt_numel, 0);
  already_match_indicator[1].resize(clt_numel, 0);

  ApplyOneKey(absl::MakeSpan(clt_keys_), absl::MakeSpan(svr_keys_),
              svr_payloads_, absl::MakeSpan(already_match_indicator[0]),
              absl::MakeSpan(already_match_indicator[1]));

  int64_t num_payloads = std::get<2>(GetParam());
  int count = 0;
  auto got_payload = spu::mpc::ring_add(ret_payload_[0], ret_payload_[1]);

  for (int64_t i = 0; i < clt_numel; ++i) {
    if (ret_eq_bit_[0][i] ^ ret_eq_bit_[1][i]) {
      int j = matched_pairs_[count].second;
      for (int64_t b = 0; b < num_payloads; ++b) {
        ASSERT_EQ(got_payload.at<int64_t>({i, b}),
                  svr_payloads_.at<int64_t>({j, b}));
      }
      count += 1;
    }
  }
  ASSERT_EQ(count, matched_pairs_.size());
}

TEST_P(CPSITest, MultipleKeys) {
  int64_t svr_numel = std::get<0>(GetParam());
  int64_t clt_numel = std::get<1>(GetParam());
  int64_t num_payloads = std::get<2>(GetParam());

  if (svr_numel < clt_numel) {
    return;
  }

  std::vector<uint8_t> already_match_indicator[2];

  already_match_indicator[0].resize(clt_numel, 0);
  already_match_indicator[1].resize(clt_numel, 0);

  spu::NdArrayRef acc_payload[2];
  ApplyOneKey(absl::MakeSpan(clt_keys_), absl::MakeSpan(svr_keys_),
              svr_payloads_, absl::MakeSpan(already_match_indicator[0]),
              absl::MakeSpan(already_match_indicator[1]));
  acc_payload[0] = ret_payload_[0];
  acc_payload[1] = ret_payload_[1];

  ApplyOneKey(absl::MakeSpan(clt_2nd_keys_), absl::MakeSpan(svr_2nd_keys_),
              svr_payloads_, absl::MakeSpan(already_match_indicator[0]),
              absl::MakeSpan(already_match_indicator[1]));
  acc_payload[0] = spu::mpc::ring_add(acc_payload[0], ret_payload_[0]);
  acc_payload[1] = spu::mpc::ring_add(acc_payload[1], ret_payload_[1]);

  auto acc = spu::mpc::ring_add(acc_payload[0], acc_payload[1]);

  std::unordered_set<int32_t> already_matched;

  // Match on first key
  for (auto kv : matched_pairs_) {
    int32_t i = kv.first;
    int32_t j = kv.second;
    for (int64_t b = 0; b < num_payloads; ++b) {
      auto expected = svr_payloads_.at<int64_t>({j, b});
      auto got = acc.at<int64_t>({i, b});
      ASSERT_EQ(expected, got);
    }
    already_matched.insert(i);
  }

  // Match on second key
  for (auto kv : matched_2nd_pairs_) {
    int32_t i = kv.first;
    int32_t j = kv.second;

    if (already_matched.count(i)) {
      // position that already matched by the first key
      for (int64_t b = 0; b < num_payloads; ++b) {
        auto expected = svr_payloads_.at<int64_t>({j, b});
        auto got = acc.at<int64_t>({i, b});
        ASSERT_NE(expected, got);
      }
    } else {
      // position that is not matched by the first key yet
      for (int64_t b = 0; b < num_payloads; ++b) {
        auto expected = svr_payloads_.at<int64_t>({j, b});
        auto got = acc.at<int64_t>({i, b});
        ASSERT_EQ(expected, got);
      }
      already_matched.insert(i);
    }
  }

  // Count all match positions
  for (int64_t i = 0; i < clt_numel; ++i) {
    if (already_match_indicator[0][i] ^ already_match_indicator[1][i]) {
      ASSERT_EQ(already_matched.count(i), 1);
    } else {
      ASSERT_EQ(already_matched.count(i), 0);
    }
  }

  // Try to match on the same keys.
  // This should give all-zero results:
  // The already_match_indicator are unchanged
  auto ind_copy0 = already_match_indicator[0];
  auto ind_copy1 = already_match_indicator[1];
  ApplyOneKey(absl::MakeSpan(clt_keys_), absl::MakeSpan(svr_keys_),
              svr_payloads_, absl::MakeSpan(already_match_indicator[0]),
              absl::MakeSpan(already_match_indicator[1]));
  int count_eq = 0;
  for (int64_t i = 0; i < clt_numel; ++i) {
    auto before = ind_copy0[i] ^ ind_copy1[i];
    auto after = already_match_indicator[0][i] ^ already_match_indicator[1][i];
    // The boolean share might be updated
    count_eq += ind_copy0[i] == already_match_indicator[0][i];
    count_eq += ind_copy1[i] == already_match_indicator[1][i];
    // But the underlying indicator is unchanged.
    ASSERT_EQ(before, after);
  }
  EXPECT_NEAR(count_eq * 0.5 / clt_numel, 0.5, 0.1);
  // The newly computed payloads are all zero
  auto all_zero = spu::mpc::ring_add(ret_payload_[0], ret_payload_[1]);
  ASSERT_TRUE(std::all_of(
      all_zero.data<uint8_t>(),
      all_zero.data<uint8_t>() + all_zero.numel() * all_zero.elsize(),
      [](uint8_t x) { return x == 0; }));
}

}  // namespace psi::test
