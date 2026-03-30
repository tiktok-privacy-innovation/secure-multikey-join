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

namespace spu::mpc::cheetah {
class BasicOTProtocols;
}

namespace psi::mpc {

class EqTProtocol {
 public:
  // REQUIRE 1 <= compare_radix <= 8.
  explicit EqTProtocol(
      const std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols>& base,
      size_t compare_radix = 4);

  ~EqTProtocol();

  spu::NdArrayRef Compute(const spu::NdArrayRef& inp, size_t bit_width = 0);

  void set_as_sender(bool flag) { is_sender_ = flag; }

 private:
  spu::NdArrayRef DoCompute(const spu::NdArrayRef& inp, size_t bit_width = 0);

  size_t compare_radix_;
  bool is_sender_{false};
  std::shared_ptr<spu::mpc::cheetah::BasicOTProtocols> basic_ot_prot_;
};

}  // namespace psi::mpc
