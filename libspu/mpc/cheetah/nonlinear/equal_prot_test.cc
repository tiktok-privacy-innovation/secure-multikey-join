// Copyright 2022 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "libspu/mpc/cheetah/nonlinear/equal_prot.h"

#include "gtest/gtest.h"
#include "yacl/utils/elapsed_timer.h"

#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {

class EqualProtTest : public ::testing::TestWithParam<FieldType> {
  void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, EqualProtTest,
    testing::Values(FieldType::FM32, FieldType::FM64, FieldType::FM128),
    [](const testing::TestParamInfo<EqualProtTest::ParamType> &p) {
      return fmt::format("{}", p.param);
    });

TEST_P(EqualProtTest, Basic) {
  size_t kWorldSize = 2;
  Shape shape = {(1L << 20)};
  FieldType field = GetParam();
  int bw = SizeOf(field) * 8;
  if (bw > 64) {
    bw = 96;
  }

  NdArrayRef inp[2];
  inp[0] = ring_rand(field, shape);
  inp[1] = ring_rand(field, shape);

  DISPATCH_ALL_FIELDS(field, [&]() {
    auto xinp0 = NdArrayView<ring2k_t>(inp[0]);
    auto xinp1 = NdArrayView<ring2k_t>(inp[1]);
    std::copy_n(&xinp1[0], 5, &xinp0[0]);
  });

  NdArrayRef eq_oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    int rank = ctx->Rank();
    auto base =
        std::make_shared<BasicOTProtocols>(conn, CheetahOtKind::YACL_Ferret);

      size_t sent = ctx->GetStats()->sent_bytes;
      size_t recv = ctx->GetStats()->recv_bytes;
      EqualProtocol eq_prot(base, bw == 64 ? 3 : 4);

      yacl::ElapsedTimer timer;
      eq_oup[rank] = eq_prot.FlattedCompute(inp[rank], bw);
      sent = ctx->GetStats()->sent_bytes - sent;
      recv = ctx->GetStats()->recv_bytes - recv;

      if (rank == 0) {
        printf("bw %d Took %fs %f MiB\n", bw, timer.CountSec(),
               (sent + recv) / 1024. / 1024.);
      }
  });

  SPU_ENFORCE_EQ(eq_oup[0].shape(), shape);
  SPU_ENFORCE_EQ(eq_oup[1].shape(), shape);

  DISPATCH_ALL_FIELDS(field, [&]() {
    auto xeq0 = NdArrayView<ring2k_t>(eq_oup[0]);
    auto xeq1 = NdArrayView<ring2k_t>(eq_oup[1]);
    auto xinp0 = NdArrayView<ring2k_t>(inp[0]);
    auto xinp1 = NdArrayView<ring2k_t>(inp[1]);

    for (int64_t i = 0; i < shape.numel(); ++i) {
      bool expected = xinp0[i] == xinp1[i];
      bool got_eq = xeq0[i] ^ xeq1[i];
      EXPECT_EQ(expected, got_eq);
    }
  });
}

TEST_P(EqualProtTest, FlattenTheTree) {
  size_t kWorldSize = 2;
  Shape shape = {1L << 20};
  FieldType field = GetParam();
  int bw = SizeOf(field) * 8;
  bw = 96;

  NdArrayRef inp[2];
  inp[0] = ring_rand(field, shape);
  inp[1] = ring_rand(field, shape);

  DISPATCH_ALL_FIELDS(field, [&]() {
    auto xinp0 = NdArrayView<ring2k_t>(inp[0]);
    auto xinp1 = NdArrayView<ring2k_t>(inp[1]);
    std::copy_n(&xinp1[0], 10000, &xinp0[0]);
  });

  NdArrayRef eq_oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    int rank = ctx->Rank();
    auto base =
        std::make_shared<BasicOTProtocols>(conn, CheetahOtKind::YACL_Ferret);

    for (int radix : {5}) {
      yacl::ElapsedTimer timer;
      size_t sent = ctx->GetStats()->sent_bytes;
      size_t recv = ctx->GetStats()->recv_bytes;

      EqualProtocol eq_prot(base, radix);
      int nrep = 1;

      for (int rep = 0; rep < nrep; ++rep) {
        eq_oup[rank] = eq_prot.FlattedCompute(inp[rank], bw);
      }
      sent = ctx->GetStats()->sent_bytes - sent;
      recv = ctx->GetStats()->recv_bytes - recv;

      if (rank == 0) {
        SPDLOG_INFO("FlattenEqual radix {}, {} MiB per {} ms", radix,
                    (sent + recv) / 1024. / 1024., timer.CountMs());
      }
    }
  });

  SPU_ENFORCE_EQ(eq_oup[0].shape(), shape);
  SPU_ENFORCE_EQ(eq_oup[1].shape(), shape);

  DISPATCH_ALL_FIELDS(field, [&]() {
    auto xeq0 = NdArrayView<ring2k_t>(eq_oup[0]);
    auto xeq1 = NdArrayView<ring2k_t>(eq_oup[1]);
    auto xinp0 = NdArrayView<ring2k_t>(inp[0]);
    auto xinp1 = NdArrayView<ring2k_t>(inp[1]);

    for (int64_t i = 0; i < shape.numel(); ++i) {
      auto expected = xinp0[i] == xinp1[i];
      auto got_eq = xeq0[i] ^ xeq1[i];
      EXPECT_EQ(expected, got_eq);
    }
  });
}

TEST_P(EqualProtTest, EqualWithBitWidth) {
  size_t kWorldSize = 2;
  Shape shape = {1L << 21};
  FieldType field = GetParam();
  int64_t bw = 7;

  NdArrayRef inp[2];
  inp[0] = ring_rand(field, shape);
  inp[1] = ring_rand(field, shape);

  DISPATCH_ALL_FIELDS(field, [&]() {
    auto xinp0 = NdArrayView<ring2k_t>(inp[0]);
    auto xinp1 = NdArrayView<ring2k_t>(inp[1]);
    std::copy_n(&xinp1[0], 4, &xinp0[0]);
  });

  DISPATCH_ALL_FIELDS(field, [&]() {
    ring2k_t mask = (static_cast<ring2k_t>(1) << bw) - 1;
    for (int idx : {0, 1}) {
      auto xinp = NdArrayView<ring2k_t>(inp[idx]);
      pforeach(0, inp[idx].numel(), [&](int64_t i) { xinp[i] &= mask; });
    }
  });

  NdArrayRef eq_oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    int rank = ctx->Rank();
    auto base =
        std::make_shared<BasicOTProtocols>(conn, CheetahOtKind::YACL_Ferret);
    EqualProtocol eq_prot(base, /*radix*/ bw);

    yacl::ElapsedTimer timer;
    eq_oup[rank] = eq_prot.Compute(inp[rank], /*bit_width*/ bw);
    if (rank == 0) {
      printf("Took %fms\n", timer.CountMs());
    }
  });

  SPU_ENFORCE_EQ(eq_oup[0].shape(), shape);
  SPU_ENFORCE_EQ(eq_oup[1].shape(), shape);

  DISPATCH_ALL_FIELDS(field, [&]() {
    auto xeq0 = NdArrayView<ring2k_t>(eq_oup[0]);
    auto xeq1 = NdArrayView<ring2k_t>(eq_oup[1]);
    auto xinp0 = NdArrayView<ring2k_t>(inp[0]);
    auto xinp1 = NdArrayView<ring2k_t>(inp[1]);

    for (int64_t i = 0; i < shape.numel(); ++i) {
      bool expected = xinp0[i] == xinp1[i];
      bool got_eq = xeq0[i] ^ xeq1[i];
      EXPECT_EQ(expected, got_eq);
    }
  });
}

TEST_P(EqualProtTest, Batch) {
  size_t kWorldSize = 2;
  Shape shape = {1L << 15};
  FieldType field = GetParam();

  [[maybe_unused]] int64_t fw = SizeOf(field) * 8;
  auto bw = fw;
  if (field == FM128) {
    bw = 96;
  }
  [[maybe_unused]] int64_t batch = 128;
  Shape bshape = {shape.numel(), batch};

  NdArrayRef inp[2];
  inp[0] = ring_rand(field, bshape);
  inp[1] = ring_rand(field, shape);

  ring_rshift_(inp[0], {fw - 8});
  ring_rshift_(inp[1], {fw - 8});

  // a0, a1, a2, ..., aB
  // b
  // eq0, eq1, eq2, ..., eqB
  NdArrayRef eq_oup[2];
  utils::simulate(kWorldSize, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    int rank = ctx->Rank();

    auto base =
        std::make_shared<BasicOTProtocols>(conn, CheetahOtKind::YACL_Ferret);

    EqualProtocol eq_prot(base, /*radix*/ 8);
    size_t sent = ctx->GetStats()->sent_bytes;
    yacl::ElapsedTimer timer;
    eq_oup[rank] = eq_prot.BatchCompute(inp[rank], shape.numel(), bw, batch);
    sent = ctx->GetStats()->sent_bytes - sent;
    // printf("#sent %f KiB for 1-vs-%lld (#batch = %lld) %fms\n", sent / 1024.,
    //        batch, shape.numel(), timer.CountMs());
  });

  SPU_ENFORCE_EQ(eq_oup[0].shape(), bshape);
  SPU_ENFORCE_EQ(eq_oup[1].shape(), bshape);

  DISPATCH_ALL_FIELDS(field, [&]() {
    auto xin0 = NdArrayView<ring2k_t>(inp[0]);
    auto xin1 = NdArrayView<ring2k_t>(inp[1]);
    auto xeq0 = NdArrayView<ring2k_t>(eq_oup[0]);
    auto xeq1 = NdArrayView<ring2k_t>(eq_oup[1]);

    int count_same = 0;
    for (int64_t i = 0, j = 0; i < bshape.numel(); i += batch, ++j) {
      for (int64_t k = 0; k < batch; ++k) {
        bool expected = xin0[i + k] == xin1[j];
        bool got = xeq0[i + k] ^ xeq1[i + k];
        count_same += expected ? 1 : 0;
        ASSERT_EQ(expected, got);
      }
    }

    printf("#eq %d\n", count_same);
  });
}

}  // namespace spu::mpc::cheetah
