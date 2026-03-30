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

#include "psi/permute.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>

#include "gtest/gtest.h"
#include "yacl/crypto/rand/rand.h"

#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/utils/simulate.h"

namespace psi::mpc::test {

template <typename T>
T makeBitsMask(int bw) {
  if (bw == sizeof(T) * 8) {
    return (T)(-1);
  }
  return (static_cast<T>(1) << bw) - 1;
}

template <typename T>
class PermuteTest : public ::testing::Test {};

template <typename T, int BW>
struct MyType {
  using Ty = T;
  static constexpr int Bw = BW;
};

struct MyU1 : MyType<uint8_t, 1> {};
struct MyU32 : MyType<uint32_t, 32> {};
struct MyU64 : MyType<uint64_t, 64> {};
struct MyU128 : MyType<uint128_t, 128> {};

using MyTypes = ::testing::Types<MyU1, MyU32, MyU64, MyU128>;

TYPED_TEST_SUITE(PermuteTest, MyTypes);

TYPED_TEST(PermuteTest, Boolean) {
  using T = typename TypeParam::Ty;
  int bw = TypeParam::Bw;
  const T mask = makeBitsMask<T>(bw);
  yacl::crypto::Prg<T> prng(12345);

  for (int numel : {2, 3, 10, 1024, 2000}) {
    std::vector<T> input[2];
    std::vector<T> output[2];
    input[0].resize(numel);
    input[1].resize(numel);
    output[0].resize(numel);
    output[1].resize(numel);

    prng.Fill(absl::MakeSpan(input[0]));
    prng.Fill(absl::MakeSpan(input[1]));
    std::transform(input[0].begin(), input[0].end(), input[0].begin(),
                   [mask](T x) { return x & mask; });
    std::transform(input[1].begin(), input[1].end(), input[1].begin(),
                   [mask](T x) { return x & mask; });

    std::vector<int32_t> perm(numel);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), std::default_random_engine(0));

    spu::mpc::utils::simulate(
        2, [&](std::shared_ptr<yacl::link::Context> conn) {
          int rank = conn->Rank();
          auto rot = std::make_shared<spu::mpc::cheetah::BasicOTProtocols>(
              conn, spu::CheetahOtKind::YACL_Ferret);
          PermuteBase::Meta meta;
          meta.is_arithmetic = false;
          meta.payload_width = sizeof(T) * 8;
          meta.numel = numel;
          if (rank == 0) {
            PermuteSender sender(meta);

            sender.Send(absl::MakeSpan(input[rank]),
                        absl::MakeSpan(output[rank]), rot);
          } else {
            PermuteReceiver receiver(meta, perm);
            receiver.Recv(absl::MakeSpan(input[rank]),
                          absl::MakeSpan(output[rank]), rot);
          }
        });
    for (int i = 0; i < numel; ++i) {
      ASSERT_EQ(input[0][perm[i]] ^ input[1][perm[i]],
                output[0][i] ^ output[1][i]);
    }
  }
}

TYPED_TEST(PermuteTest, Arith) {
  using T = typename TypeParam::Ty;
  const int bw = TypeParam::Bw;
  const T mask = makeBitsMask<T>(bw);

  if (bw == 1) {
    return;
  }
  yacl::crypto::Prg<T> prng(12345);

  for (int numel : {2, 3, 10, 1024, 2000}) {
    std::vector<T> input[2];
    std::vector<T> output[2];
    input[0].resize(numel);
    input[1].resize(numel);
    output[0].resize(numel);
    output[1].resize(numel);

    prng.Fill(absl::MakeSpan(input[0]));
    prng.Fill(absl::MakeSpan(input[1]));
    std::transform(input[0].begin(), input[0].end(), input[0].begin(),
                   [mask](T x) { return x & mask; });
    std::transform(input[1].begin(), input[1].end(), input[1].begin(),
                   [mask](T x) { return x & mask; });
    std::vector<int32_t> perm(numel);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), std::default_random_engine(0));

    spu::mpc::utils::simulate(
        2, [&](std::shared_ptr<yacl::link::Context> conn) {
          int rank = conn->Rank();
          auto rot = std::make_shared<spu::mpc::cheetah::BasicOTProtocols>(
              conn, spu::CheetahOtKind::YACL_Ferret);
          PermuteBase::Meta meta;
          meta.is_arithmetic = true;
          meta.payload_width = sizeof(T) * 8;
          meta.numel = numel;
          if (rank == 0) {
            PermuteSender sender(meta);

            sender.Send(absl::MakeSpan(input[rank]),
                        absl::MakeSpan(output[rank]), rot);
          } else {
            PermuteReceiver receiver(meta, perm);
            receiver.Recv(absl::MakeSpan(input[rank]),
                          absl::MakeSpan(output[rank]), rot);
          }
        });

    for (int i = 0; i < numel; ++i) {
      ASSERT_EQ(input[0][perm[i]] + input[1][perm[i]],
                output[0][i] + output[1][i]);
    }
  }
}

}  // namespace psi::mpc::test
