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

#include "absl/types/span.h"
#include "yacl/base/int128.h"

#include "psi/benes_network.h"

namespace spu::mpc::cheetah {
class BasicOTProtocols;
}

namespace psi::mpc {

class PermuteBase {
 public:
  using ROT = std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols>;
  // md5("Permute-FERRET_OT-1-0-0")
  constexpr static uint64_t kPRPKey[2] = {0x0193810b5735cc87,
                                          0x2ae40a13041ccc87};
  struct Meta {
    bool is_arithmetic;
    int32_t payload_width;
    int64_t numel;
  };

  explicit PermuteBase(const Meta &meta);

  ~PermuteBase() = default;

 protected:
  Meta meta_;

  std::unique_ptr<BenesNetwork> bn_;
};

class PermuteSender : public PermuteBase {
 public:
  PermuteSender(const Meta &meta);

  ~PermuteSender() = default;
  void Send(absl::Span<const uint8_t> input, absl::Span<uint8_t> out,
            const ROT &rot) const;
  void Send(absl::Span<const uint32_t> input, absl::Span<uint32_t> out,
            const ROT &rot) const;
  void Send(absl::Span<const uint64_t> input, absl::Span<uint64_t> out,
            const ROT &rot) const;
  void Send(absl::Span<const uint128_t> input, absl::Span<uint128_t> out,
            const ROT &rot) const;
};

class PermuteReceiver : public PermuteBase {
 public:
  PermuteReceiver(const Meta &meta, absl::Span<const int32_t> perm);

  ~PermuteReceiver() = default;

  void Recv(absl::Span<const uint8_t> input, absl::Span<uint8_t> out,
            const ROT &rot) const;
  void Recv(absl::Span<const uint32_t> input, absl::Span<uint32_t> out,
            const ROT &rot) const;
  void Recv(absl::Span<const uint64_t> input, absl::Span<uint64_t> out,
            const ROT &rot) const;
  void Recv(absl::Span<const uint128_t> input, absl::Span<uint128_t> out,
            const ROT &rot) const;
};

}  // namespace psi::mpc
