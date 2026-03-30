#include <algorithm>
#include <cstdint>
#include <random>

#include "gtest/gtest.h"
#include "yacl/base/int128.h"
#include "yacl/crypto/tools/prg.h"
#include "yacl/crypto/tools/rp.h"

#include "libspu/mpc/cheetah/nonlinear/benes_network.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/common/communicator.h"
#include "libspu/mpc/utils/ring_ops.h"
#include "libspu/mpc/utils/simulate.h"

class OSNTest : public ::testing::TestWithParam<int> {};

INSTANTIATE_TEST_SUITE_P(
    Cheetah, OSNTest, testing::Values(2, 3, 4, 16, 256, 4096, 15, 88, 3456),
    [](const testing::TestParamInfo<OSNTest::ParamType>& p) {
      return fmt::format("n{}", p.param);
    });

TEST_P(OSNTest, OSN_additive) {
  using scalar_t = uint128_t;
  size_t bw = 100;
  scalar_t mod_mask = static_cast<scalar_t>(-1);
  if (bw < sizeof(scalar_t) * 8) {
    mod_mask = (static_cast<scalar_t>(1) << bw) - 1;
  }

  int64_t N = GetParam();
  yacl::crypto::Prg<scalar_t> prg;
  int64_t ln = std::ceil(std::log2(N));
  int64_t width = N / 2;
  int64_t levels = 2 * ln - 1;
  int64_t gates = levels * width;

  std::vector<int> permutation(N);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::shuffle(permutation.begin(), permutation.end(),
               std::default_random_engine(std::time(0)));

  spu::mpc::cheetah::BenesNetwork bn(permutation);
  std::vector<uint8_t> control_bits(gates);
  bn.get_control_bits(absl::MakeSpan(control_bits));
  bn.set_arithmetic_payload(bw);

  std::vector<scalar_t> values(N);
  std::generate_n(values.begin(), N, [&]() { return prg(); });
  for (auto& v : values) {
    v &= mod_mask;
  }

  auto copy = values;
  std::vector<scalar_t> send_ot_msg0(gates);
  std::vector<scalar_t> send_ot_msg1(gates);
  std::vector<scalar_t> prp_ot_msg0(gates);
  std::vector<scalar_t> prp_ot_msg1(gates);

  std::vector<scalar_t> recv_ot_msg(gates);
  std::vector<scalar_t> prp_recv_ot_msg(gates);
  yacl::crypto::RP rp(yacl::crypto::RP::Ctype::AES128_ECB, 0, 0);

  spu::mpc::utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<spu::mpc::Communicator>(ctx);
    int rank = ctx->Rank();
    auto base = std::make_shared<spu::mpc::cheetah::BasicOTProtocols>(
        conn, spu::CheetahOtKind::YACL_Ferret);
    if (rank == 0) {
      base->GetReceiverCOT()->RecvRMCC(absl::MakeConstSpan(control_bits),
                                       absl::MakeSpan(recv_ot_msg), bw);
      rp.GenForMultiInputs(absl::MakeSpan(recv_ot_msg),
                           absl::MakeSpan(prp_recv_ot_msg));
    } else {
      base->GetSenderCOT()->SendRMCC(absl::MakeSpan(send_ot_msg0),
                                     absl::MakeSpan(send_ot_msg1), bw);
      rp.GenForMultiInputs(absl::MakeSpan(&send_ot_msg0[0], gates),
                           absl::MakeSpan(&prp_ot_msg0[0], gates));
      rp.GenForMultiInputs(absl::MakeSpan(&send_ot_msg1[0], gates),
                           absl::MakeSpan(&prp_ot_msg1[0], gates));
    }
  });

  std::vector<scalar_t> corr0(gates);
  std::vector<scalar_t> corr1(gates);
  std::vector<scalar_t> mask(N, 0);
  std::generate_n(mask.begin(), N, [&]() { return prg(); });

  auto shr0 = mask;
  bn.prepare_correction(
      absl::MakeSpan(shr0), absl::MakeSpan(corr0), absl::MakeSpan(corr1),
      absl::MakeConstSpan(send_ot_msg0), absl::MakeConstSpan(send_ot_msg1),
      absl::MakeConstSpan(prp_ot_msg0), absl::MakeConstSpan(prp_ot_msg1));

  for (int64_t i = 0; i < gates; ++i) {
    if (control_bits[i]) {
      recv_ot_msg[i] += corr0[i];
      prp_recv_ot_msg[i] += corr1[i];

      recv_ot_msg[i] &= mod_mask;
      prp_recv_ot_msg[i] &= mod_mask;
    }
  }

  auto shr1 = values;
  for (int64_t i = 0; i < N; ++i) {
    shr1[i] = (values[i] - mask[i]) & mod_mask;
  }
  bn.eval_with_masks(absl::MakeSpan(shr1), absl::MakeConstSpan(recv_ot_msg),
                     absl::MakeConstSpan(prp_recv_ot_msg));

  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(copy[permutation[i]], (shr0[i] + shr1[i]) & mod_mask);
  }
}

TEST_P(OSNTest, OSN_boolean) {
  using scalar_t = uint128_t;

  int64_t N = GetParam();
  yacl::crypto::Prg<scalar_t> prg;
  int64_t ln = std::ceil(std::log2(N));
  int64_t width = N / 2;
  int64_t levels = 2 * ln - 1;
  int64_t gates = levels * width;

  std::vector<int> permutation(N);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::shuffle(permutation.begin(), permutation.end(),
               std::default_random_engine(std::time(0)));

  spu::mpc::cheetah::BenesNetwork bn(permutation);
  std::vector<uint8_t> control_bits(gates);
  bn.get_control_bits(absl::MakeSpan(control_bits));
  bn.set_boolean_payload(sizeof(scalar_t) * 8);

  std::vector<scalar_t> values(N);
  std::generate_n(values.begin(), N, [&]() { return prg(); });

  auto copy = values;
  std::vector<scalar_t> ot_msg0(gates);
  std::vector<scalar_t> ot_msg1(gates);
  std::vector<scalar_t> prp_ot_msg0(gates);
  std::vector<scalar_t> prp_ot_msg1(gates);

  std::vector<scalar_t> recv_ot_msg(gates);
  std::vector<scalar_t> prp_recv_ot_msg(gates);
  yacl::crypto::RP rp(yacl::crypto::RP::Ctype::AES128_ECB, 0, 0);

  spu::mpc::utils::simulate(2, [&](std::shared_ptr<yacl::link::Context> ctx) {
    auto conn = std::make_shared<spu::mpc::Communicator>(ctx);
    int rank = ctx->Rank();
    auto base = std::make_shared<spu::mpc::cheetah::BasicOTProtocols>(
        conn, spu::CheetahOtKind::YACL_Ferret);
    if (rank == 0) {
      base->GetReceiverCOT()->RecvRMCC(absl::MakeConstSpan(control_bits),
                                       absl::MakeSpan(recv_ot_msg));
      rp.GenForMultiInputs(absl::MakeSpan(recv_ot_msg),
                           absl::MakeSpan(prp_recv_ot_msg));
    } else {
      base->GetSenderCOT()->SendRMCC(absl::MakeSpan(ot_msg0),
                                     absl::MakeSpan(ot_msg1));

      rp.GenForMultiInputs(absl::MakeSpan(&ot_msg0[0], gates),
                           absl::MakeSpan(&prp_ot_msg0[0], gates));
      rp.GenForMultiInputs(absl::MakeSpan(&ot_msg1[0], gates),
                           absl::MakeSpan(&prp_ot_msg1[0], gates));
    }
  });

  std::vector<scalar_t> corr0(gates);
  std::vector<scalar_t> corr1(gates);
  std::vector<scalar_t> mask(N, 0);
  std::generate_n(mask.begin(), N, [&]() { return prg(); });

  auto shr0 = mask;
  bn.prepare_correction(
      absl::MakeSpan(shr0), absl::MakeSpan(corr0), absl::MakeSpan(corr1),
      absl::MakeConstSpan(ot_msg0), absl::MakeConstSpan(ot_msg1),
      absl::MakeConstSpan(prp_ot_msg0), absl::MakeConstSpan(prp_ot_msg1));

  for (int64_t i = 0; i < gates; ++i) {
    if (control_bits[i]) {
      recv_ot_msg[i] ^= corr0[i];
      prp_recv_ot_msg[i] ^= corr1[i];
    }
  }

  auto shr1 = values;
  for (int64_t i = 0; i < N; ++i) {
    shr1[i] = values[i] ^ mask[i];
  }
  bn.eval_with_masks(absl::MakeSpan(shr1), absl::MakeConstSpan(recv_ot_msg),
                     absl::MakeConstSpan(prp_recv_ot_msg));

  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(copy[permutation[i]], shr0[i] ^ shr1[i]);
  }
}
