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

#include "libspu/core/ndarray_ref.h"

#include "psi/opprf.h"

namespace spu::mpc::cheetah {
class BasicOTProtocols;

}  // namespace spu::mpc::cheetah

namespace psi {

using OTProt = std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols>;

class CircuitPSIBase {
 public:
  CircuitPSIBase(std::array<OTProt, 2> ot_pair);

  spu::NdArrayRef WaterFallUpdateInplace(
      absl::Span<uint8_t> already_match_indicator,
      absl::Span<const uint8_t> this_key_match_indicator,
      const spu::NdArrayRef& this_key_match_payload) const;

 protected:
  std::vector<uint8_t> EqualTest(spu::NdArrayRef x, int64_t bit_width) const;

  int rank_;
  OTProt ot_;
  OTProt duplx_ot_;
};

class CircuitPSIServer : public CircuitPSIBase {
 public:
  CircuitPSIServer(std::array<OTProt, 2> ot_pair);

  std::tuple<std::vector<uint8_t>, spu::NdArrayRef> Send(
      absl::Span<const uint128_t> keys, const spu::NdArrayRef& payload,
      int64_t client_numel,
      const std::shared_ptr<yacl::link::Context>& conn) const;

 private:
  void PermuteInplace(absl::Span<uint8_t> eq_bits) const;

  spu::NdArrayRef PermuteTranspose(const spu::NdArrayRef& payload,
                                   int64_t keep_front_size) const;
};

class CircuitPSIClient : public CircuitPSIBase {
 public:
  CircuitPSIClient(std::array<OTProt, 2> ot_pair);

  std::tuple<std::vector<uint8_t>, spu::NdArrayRef> Recv(
      absl::Span<const uint128_t> keys, const spu::Shape& svr_payload_shape,
      const std::shared_ptr<yacl::link::Context>& conn) const;

 private:
  void PermuteInplace(absl::Span<uint8_t> eq_bits,
                      absl::Span<const int32_t> perm) const;

  spu::NdArrayRef PermuteTranspose(const spu::NdArrayRef& payload,
                                   int64_t keep_front_size,
                                   absl::Span<const int32_t> perm) const;
};

}  // namespace psi
