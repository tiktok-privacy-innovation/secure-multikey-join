#include "libspu/mpc/cheetah/nonlinear/osn_prot.h"

#include <random>

#include "gtest/gtest.h"
#include "yacl/utils/elapsed_timer.h"

#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

namespace spu::mpc::cheetah {

class OSNProtTest
    : public ::testing::TestWithParam<std::tuple<FieldType, int>> {};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, OSNProtTest,
    testing::Combine(testing::Values(FieldType::FM32, FieldType::FM64,
                                     FieldType::FM128),
                     testing::Values(1L << 14, 1L << 17, 1L << 20)),
    [](const testing::TestParamInfo<OSNProtTest::ParamType>& p) {
      return fmt::format("Ft{}n{}", std::get<0>(p.param), std::get<1>(p.param));
    });

TEST_P(OSNProtTest, Boolean) {
  FieldType ft = std::get<0>(GetParam());
  int64_t numel = std::get<1>(GetParam());
  OSNProtocol::Meta meta;
  meta.numel = numel;
  meta.is_arithmetic = false;
  meta.payload_width = 1;
  meta.payload_type = ft;

  std::vector<int32_t> permutation(numel);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::shuffle(permutation.begin(), permutation.end(),
               std::default_random_engine(std::time(0)));

  auto input = ring_rand(ft, {numel});
  DISPATCH_ALL_FIELDS(ft, [&]() {
    NdArrayView<ring2k_t> xgot(input);
    ring2k_t mask = static_cast<ring2k_t>(-1);
    if (meta.payload_width < (int)sizeof(ring2k_t) * 8) {
      mask = (static_cast<ring2k_t>(1) << meta.payload_width) - 1;
    }
    for (int64_t i = 0; i < numel; ++i) {
      xgot[i] &= mask;
    }
  });

  NdArrayRef oup[2];
  utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    int rank = ctx->Rank();
    auto base =
        std::make_shared<BasicOTProtocols>(conn, CheetahOtKind::YACL_Ferret);

    if (rank == 0) {
      size_t sent = ctx->GetStats()->sent_bytes;
      size_t recv = ctx->GetStats()->recv_bytes;

      yacl::ElapsedTimer timer;
      OSNProtocol osn(meta, permutation);
      oup[rank] = osn.Recv(numel, base);

      sent = ctx->GetStats()->sent_bytes - sent;
      recv = ctx->GetStats()->recv_bytes - recv;

      SPDLOG_INFO("osn {}, {} MiB, {} sec", numel,
                  (sent + recv) / 1024.0 / 1024.0, timer.CountSec());
    } else {
      OSNProtocol osn(meta);
      oup[rank] = osn.Send(input, base);
    }
  });

  auto got = ring_xor(oup[0], oup[1]);
  DISPATCH_ALL_FIELDS(ft, [&]() {
    NdArrayView<ring2k_t> xgot(got);
    NdArrayView<ring2k_t> xexpected(input);
    for (int64_t i = 0; i < numel; ++i) {
      ASSERT_EQ(xgot[i], xexpected[permutation[i]]);
    }
  });
}

TEST_P(OSNProtTest, Arith) {
  FieldType ft = std::get<0>(GetParam());
  int64_t numel = std::get<1>(GetParam());
  OSNProtocol::Meta meta;
  meta.numel = numel;
  meta.is_arithmetic = true;
  meta.payload_width = SizeOf(ft) * 8;
  meta.payload_type = ft;

  std::vector<int32_t> permutation(numel);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::shuffle(permutation.begin(), permutation.end(),
               std::default_random_engine(std::time(0)));
  auto input = ring_rand(ft, {numel});

  NdArrayRef oup[2];
  utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<Communicator>(ctx);
    int rank = ctx->Rank();
    auto base =
        std::make_shared<BasicOTProtocols>(conn, CheetahOtKind::YACL_Ferret);
    if (rank == 0) {
      size_t sent = ctx->GetStats()->sent_bytes;
      size_t recv = ctx->GetStats()->recv_bytes;

      OSNProtocol osn(meta, permutation);

      yacl::ElapsedTimer timer;
      for (size_t rep = 0; rep < 5; ++rep) {
        oup[rank] = osn.Recv(numel, base);
      }

      sent = ctx->GetStats()->sent_bytes - sent;
      recv = ctx->GetStats()->recv_bytes - recv;

      SPDLOG_INFO("osn {}, {} MiB, {} sec", numel,
                  (sent + recv) / 1024.0 / 1024.0 / 5,
                  timer.CountMs() / 1e3 / 5);
    } else {
      OSNProtocol osn(meta);

      for (size_t rep = 0; rep < 5; ++rep) {
        oup[rank] = osn.Send(input, base);
      }
    }
  });

  auto got = ring_add(oup[0], oup[1]);
  DISPATCH_ALL_FIELDS(ft, [&]() {
    NdArrayView<ring2k_t> xgot(got);
    NdArrayView<ring2k_t> xexpected(input);
    for (int64_t i = 0; i < numel; ++i) {
      ASSERT_EQ(xgot[i], xexpected[permutation[i]]);
    }
  });
}

// TEST_P(OSNProtTest, Boolean_Shared) {
//   FieldType ft = std::get<0>(GetParam());
//   int64_t numel = std::get<1>(GetParam());
//   OSNProtocol::Meta meta;
//   meta.numel = numel;
//   meta.is_arithmetic = false;
//   meta.payload_width = SizeOf(ft) * 8;
//   meta.payload_type = ft;

//   std::vector<int32_t> permutation(numel);
//   std::iota(permutation.begin(), permutation.end(), 0);
//   std::shuffle(permutation.begin(), permutation.end(),
//                std::default_random_engine(std::time(0)));

//   NdArrayRef inp[2];
//   inp[0] = ring_rand(ft, {numel});
//   inp[1] = ring_rand(ft, {numel});

//   NdArrayRef oup[2];
//   utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> ctx) {
//     auto conn = std::make_shared<Communicator>(ctx);
//     int rank = ctx->Rank();
//     auto base =
//         std::make_shared<BasicOTProtocols>(conn, CheetahOtKind::YACL_Ferret);
//     if (rank == 0) {
//       OSNProtocol osn(meta, permutation);
//       oup[rank] = osn.Recv(inp[rank], base);
//     } else {
//       size_t sent = ctx->GetStats()->sent_bytes;
//       size_t recv = ctx->GetStats()->recv_bytes;
//       OSNProtocol osn(meta);
//       oup[rank] = osn.Send(inp[rank], base);

//       sent = ctx->GetStats()->sent_bytes - sent;
//       recv = ctx->GetStats()->recv_bytes - recv;
//       SPDLOG_INFO("osn {}, {} MiB", numel, (sent + recv) / 1024.0 / 1024.0);
//     }
//   });

//   auto input = ring_xor(inp[0], inp[1]);
//   auto got = ring_xor(oup[0], oup[1]);
//   DISPATCH_ALL_FIELDS(ft, [&]() {
//     NdArrayView<ring2k_t> xgot(got);
//     NdArrayView<ring2k_t> xexpected(input);
//     for (int64_t i = 0; i < numel; ++i) {
//       ASSERT_EQ(xgot[i], xexpected[permutation[i]]);
//     }
//   });
// }

// TEST_P(OSNProtTest, Arith_Shared) {
//   FieldType ft = std::get<0>(GetParam());
//   int64_t numel = std::get<1>(GetParam());
//   OSNProtocol::Meta meta;
//   meta.numel = numel;
//   meta.is_arithmetic = true;
//   meta.payload_width = SizeOf(ft) * 8;
//   meta.payload_type = ft;

//   std::vector<int32_t> permutation(numel);
//   std::iota(permutation.begin(), permutation.end(), 0);
//   std::shuffle(permutation.begin(), permutation.end(),
//                std::default_random_engine(std::time(0)));

//   NdArrayRef inp[2];
//   inp[0] = ring_rand(ft, {numel});
//   inp[1] = ring_rand(ft, {numel});

//   NdArrayRef oup[2];
//   utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> ctx) {
//     auto conn = std::make_shared<Communicator>(ctx);
//     int rank = ctx->Rank();
//     auto base =
//         std::make_shared<BasicOTProtocols>(conn, CheetahOtKind::YACL_Ferret);
//     if (rank == 0) {
//       OSNProtocol osn(meta, permutation);
//       oup[rank] = osn.Recv(inp[rank], base);
//     } else {
//       size_t sent = ctx->GetStats()->sent_bytes;
//       size_t recv = ctx->GetStats()->recv_bytes;
//       OSNProtocol osn(meta);
//       oup[rank] = osn.Send(inp[rank], base);

//       sent = ctx->GetStats()->sent_bytes - sent;
//       recv = ctx->GetStats()->recv_bytes - recv;
//       SPDLOG_INFO("osn {}, {} MiB", numel, (sent + recv) / 1024.0 / 1024.0);
//     }
//   });

//   auto input = ring_add(inp[0], inp[1]);
//   auto got = ring_add(oup[0], oup[1]);
//   DISPATCH_ALL_FIELDS(ft, [&]() {
//     NdArrayView<ring2k_t> xgot(got);
//     NdArrayView<ring2k_t> xexpected(input);
//     for (int64_t i = 0; i < numel; ++i) {
//       ASSERT_EQ(xgot[i], xexpected[permutation[i]]);
//     }
//   });
// }

// TEST_P(OSNProtTest, RandomPermute) {
//   FieldType ft = std::get<0>(GetParam());
//   int64_t numel = std::get<1>(GetParam());
//   OSNProtocol::Meta meta;
//   meta.numel = numel;
//   meta.is_arithmetic = true;
//   meta.payload_width = SizeOf(ft) * 8;
//   meta.payload_type = ft;

//   std::vector<int32_t> perm0(numel);
//   std::vector<int32_t> perm1(numel);
//   std::iota(perm0.begin(), perm0.end(), 0);
//   std::iota(perm1.begin(), perm1.end(), 0);
//   std::shuffle(perm0.begin(), perm0.end(),
//                std::default_random_engine(std::time(0)));
//   std::shuffle(perm1.begin(), perm1.end(),
//                std::default_random_engine(std::time(0)));

//   NdArrayRef inp[2];
//   inp[0] = ring_rand(ft, {numel});
//   inp[1] = ring_rand(ft, {numel});

//   NdArrayRef oup[2];
//   utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> ctx) {
//     auto conn = std::make_shared<Communicator>(ctx);
//     int rank = ctx->Rank();
//     auto base =
//         std::make_shared<BasicOTProtocols>(conn, CheetahOtKind::YACL_Ferret);

//     if (rank == 0) {
//       OSNProtocol recv_osn(meta, perm0);
//       OSNProtocol send_osn(meta);

//       size_t sent = ctx->GetStats()->sent_bytes;
//       size_t recv = ctx->GetStats()->recv_bytes;
//       auto tmp = send_osn.Send(inp[rank], base);
//       oup[rank] = recv_osn.Recv(tmp, base);
//       sent = ctx->GetStats()->sent_bytes - sent;
//       recv = ctx->GetStats()->recv_bytes - recv;
//       SPDLOG_INFO("permutation {} MiB", (sent + recv) / 1024. / 1024.);
//     } else {
//       OSNProtocol recv_osn(meta, perm1);
//       OSNProtocol send_osn(meta);
//       auto tmp = recv_osn.Recv(inp[rank], base);
//       oup[rank] = send_osn.Send(tmp, base);
//     }
//   });

//   auto input = ring_add(inp[0], inp[1]);
//   auto got = ring_add(oup[0], oup[1]);
//   DISPATCH_ALL_FIELDS(ft, [&]() {
//     NdArrayView<ring2k_t> xgot(got);
//     NdArrayView<ring2k_t> xinp(input);
//     for (int64_t i = 0; i < numel; ++i) {
//       auto idx = perm0[perm1[i]];
//       ASSERT_EQ(xgot[i], xinp[idx]);

//       idx = perm1[perm0[i]];
//       ASSERT_EQ(xgot[i], xinp[idx]);
//     }
//   });
// }

// TEST_P(OSNProtTest, RandomPermute_Duplx) {
//   FieldType ft = std::get<0>(GetParam());
//   int64_t numel = std::get<1>(GetParam());
//   OSNProtocol::Meta meta;
//   meta.numel = numel;
//   meta.is_arithmetic = true;
//   meta.payload_width = SizeOf(ft) * 8;
//   meta.payload_type = ft;

//   std::vector<int32_t> perm0(numel);
//   std::vector<int32_t> perm1(numel);
//   std::iota(perm0.begin(), perm0.end(), 0);
//   std::iota(perm1.begin(), perm1.end(), 0);
//   std::shuffle(perm0.begin(), perm0.end(),
//                std::default_random_engine(std::time(0)));
//   std::shuffle(perm1.begin(), perm1.end(),
//                std::default_random_engine(std::time(0)));

//   NdArrayRef inp[2];
//   Shape shape = {numel};
//   inp[0] = ring_rand(ft, shape);
//   inp[1] = ring_rand(ft, shape);

//   NdArrayRef oup[2];
//   utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> ctx) {
//     int rank = ctx->Rank();
//     auto conn = std::make_shared<Communicator>(ctx);
//     auto conn2 = std::make_shared<Communicator>(ctx->Spawn());
//     auto base =
//         std::make_shared<BasicOTProtocols>(conn, CheetahOtKind::YACL_Ferret);
//     auto base2 =
//         std::make_shared<BasicOTProtocols>(conn2,
//         CheetahOtKind::YACL_Ferret);

//     if (rank == 0) {
//       OSNProtocol recv_osn(meta, perm0);
//       OSNProtocol send_osn(meta);

//       size_t sent = ctx->GetStats()->sent_bytes;
//       size_t recv = ctx->GetStats()->recv_bytes;
//       auto subtask = std::async([&]() -> std::array<spu::NdArrayRef, 2> {
//         return send_osn.SendPartial(shape, base2);
//       });

//       auto tmp = recv_osn.Recv(inp[rank], base);
//       auto [mask, msg] = subtask.get();
//       oup[rank] = msg;

//       send_osn.FinishSend(tmp, mask, base2);

//       sent = ctx->GetStats()->sent_bytes - sent;
//       recv = ctx->GetStats()->recv_bytes - recv;
//       SPDLOG_INFO("duplx permutation {} MiB", (sent + recv) / 1024. / 1024.);
//     } else {
//       OSNProtocol recv_osn(meta, perm1);
//       OSNProtocol send_osn(meta);

//       auto subtask =
//           std::async([&]() { return recv_osn.RecvPartial(shape, base2); });

//       auto tmp = recv_osn.Send(inp[rank], base);
//       auto recv_ot = subtask.get();

//       oup[rank] = recv_osn.FinishRecv(tmp, recv_ot, base2);
//     }
//   });

//   auto input = ring_add(inp[0], inp[1]);
//   auto got = ring_add(oup[0], oup[1]);
//   DISPATCH_ALL_FIELDS(ft, [&]() {
//     NdArrayView<ring2k_t> xgot(got);
//     NdArrayView<ring2k_t> xinp(input);
//     for (int64_t i = 0; i < numel; ++i) {
//       auto idx = perm0[perm1[i]];
//       ASSERT_EQ(xgot[i], xinp[idx]);

//       idx = perm1[perm0[i]];
//       ASSERT_EQ(xgot[i], xinp[idx]);
//     }
//   });
// }

// TEST_P(OSNProtTest, InvPermutation) {
//   FieldType ft = std::get<0>(GetParam());
//   int64_t numel = std::get<1>(GetParam());
//   OSNProtocol::Meta meta;
//   meta.numel = numel;
//   meta.is_arithmetic = true;
//   meta.payload_width = SizeOf(ft) * 8;
//   meta.payload_type = ft;

//   std::vector<int32_t> permutation(numel);
//   std::iota(permutation.begin(), permutation.end(), 0);
//   std::shuffle(permutation.begin(), permutation.end(),
//                std::default_random_engine(std::time(0)));
//   auto input = ring_rand(ft, {numel});

//   NdArrayRef oup[2];
//   utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> ctx) {
//     auto conn = std::make_shared<Communicator>(ctx);
//     int rank = ctx->Rank();
//     auto base =
//         std::make_shared<BasicOTProtocols>(conn, CheetahOtKind::YACL_Ferret);
//     if (rank == 0) {
//       size_t sent = ctx->GetStats()->sent_bytes;
//       size_t recv = ctx->GetStats()->recv_bytes;

//       OSNProtocol osn(meta, permutation);

//       oup[rank] = osn.Recv(numel, base);

//       sent = ctx->GetStats()->sent_bytes - sent;
//       recv = ctx->GetStats()->recv_bytes - recv;

//       SPDLOG_INFO("osn {}, send {} recv {} KiB", numel, (sent) / 1024.0,
//                   recv / 1024.0);
//     } else {
//       OSNProtocol osn(meta);
//       oup[rank] = osn.Send(input, base);
//     }
//   });

//   auto got = ring_add(oup[0], oup[1]);
//   DISPATCH_ALL_FIELDS(ft, [&]() {
//     NdArrayView<ring2k_t> xgot(got);
//     NdArrayView<ring2k_t> xexpected(input);
//     for (int64_t i = 0; i < numel; ++i) {
//       ASSERT_EQ(xgot[i], xexpected[permutation[i]]);
//     }
//   });
// }

}  // namespace spu::mpc::cheetah
