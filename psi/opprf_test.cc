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

#include "gtest/gtest.h"
#include "yacl/crypto/rand/rand.h"
#include "yacl/utils/elapsed_timer.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace psi::opprf::test {

class OPPRFTest
    : public testing::TestWithParam<std::tuple<int64_t, int64_t, int64_t>> {
 public:
  void SetUp() override {
    int64_t svr_numel = std::get<0>(GetParam());
    int64_t clt_numel = std::get<1>(GetParam());
    const int64_t batch = std::get<2>(GetParam());

    int64_t match_sze = std::max(1LL, std::min(svr_numel, clt_numel) / 8);
    matched_index_.resize(match_sze);
    std::vector<uint32_t> matched(clt_numel);
    std::iota(matched.begin(), matched.end(), 0);
    std::shuffle(matched.begin(), matched.end(), std::default_random_engine(0));
    std::copy_n(matched.begin(), match_sze, matched_index_.data());

    svr_keys_.resize(svr_numel);
    svr_payloads_ = spu::mpc::ring_rand(spu::FM128, {batch, svr_numel});
    clt_keys_.resize(clt_numel);

    yacl::crypto::Prg<uint128_t> prng(12345);
    prng.Fill(absl::MakeSpan(svr_keys_));
    for (int64_t i = 0; i < match_sze; ++i) {
      clt_keys_[i] = svr_keys_[matched_index_[i]];
    }
    prng.Fill(absl::MakeSpan(clt_keys_).subspan(match_sze));
  }

  std::vector<uint128_t> svr_keys_;
  spu::NdArrayRef svr_payloads_;
  std::vector<uint128_t> clt_keys_;
  std::vector<uint32_t> matched_index_;
};

INSTANTIATE_TEST_SUITE_P(
    PSI, OPPRFTest,
    testing::Combine(testing::Values(1 << 15, 1L << 20),
                     testing::Values(1 << 10, 1 << 15),
                     testing::Values(1, 2, 3)),
    [](const testing::TestParamInfo<OPPRFTest::ParamType>& p) {
      return fmt::format("Server{}Client{}Payload{}", std::get<0>(p.param),
                         std::get<1>(p.param), std::get<2>(p.param));
    });

TEST_P(OPPRFTest, Basic) {
  int64_t svr_numel = std::get<0>(GetParam());
  int64_t clt_numel = std::get<1>(GetParam());
  const int64_t batch = std::get<2>(GetParam());
  ASSERT_GE(svr_numel, clt_numel);

  auto config = DefaultOpprfConfig();
  config.num_threads = 1;
  std::vector<uint128_t> out(clt_numel * batch);

  spu::mpc::utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> conn) {
    int rank = conn->Rank();
    if (rank == 0) {
      yacl::ElapsedTimer timer;
      size_t sent = conn->GetStats()->sent_bytes;
      size_t recv = conn->GetStats()->recv_bytes;

      OPPRFSender sender(config, svr_numel, clt_numel, conn);

      sender.SendBatch(absl::MakeSpan(svr_keys_),
                       absl::MakeSpan(svr_payloads_.data<uint128_t>(),
                                      svr_payloads_.numel()),
                       batch, conn);

      sent = conn->GetStats()->sent_bytes - sent;
      recv = conn->GetStats()->recv_bytes - recv;
      double time = timer.CountMs();
      SPDLOG_INFO(
          "OPPRF Server {}, Client {} elements, {} GF(128) payloads exchange "
          "{} "
          "KiB, {} ms",
          svr_numel, clt_numel, batch, (sent + recv) / 1024., time);
    } else {
      OPPRFReceiver receiver(config, svr_numel, clt_numel, conn);
      receiver.RecvBatch(absl::MakeSpan(clt_keys_), batch, conn,
                         absl::MakeSpan(out));
    }
  });

  for (size_t i = 0; i < matched_index_.size(); ++i) {
    for (int64_t b = 0; b < batch; ++b) {
      ASSERT_EQ(out[b * clt_numel + i],
                svr_payloads_.at<uint128_t>({b, (int64_t)matched_index_[i]}));
    }
  }

  spu::NdArrayView<uint128_t> svr_payloads(svr_payloads_);
  std::sort(&svr_payloads[0], &svr_payloads[0] + svr_payloads.numel());

  // barely match here
  for (int64_t i = matched_index_.size(); i < clt_numel; ++i) {
    ASSERT_FALSE(std::binary_search(
        &svr_payloads[0], &svr_payloads[0] + svr_payloads.numel(), out[i]));
    for (int64_t b = 1; b < batch; ++b) {
      ASSERT_FALSE(std::binary_search(&svr_payloads[0],
                                      &svr_payloads[0] + svr_payloads.numel(),
                                      out[b * clt_numel + i]));
    }
  }
}

}  // namespace psi::opprf::test
