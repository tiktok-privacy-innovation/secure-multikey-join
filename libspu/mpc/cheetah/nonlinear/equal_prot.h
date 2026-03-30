// Copyright 2021 Ant Group Co., Ltd.
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

#pragma once

#include <memory>

#include "libspu/core/ndarray_ref.h"

namespace spu::mpc::cheetah {

class BasicOTProtocols;

// REF: CrypTFlow2: Practical 2-party secure inference.
// [1{x = y}]_B <- EQ(x, y) for two private input
//
// Math:
//   1. break into digits:
//       x = x0 || x1 || ..., ||xd
//       y = y0 || y1 || ..., ||yd
//      where 0 <= xi,yi < 2^{radix}
//   2. Use 1-of-2^{radix} OTs to compute the bit eq_i = [1{xi = yi}]_B
//   3. Tree-based AND 1{x = y} = AND_i eq_i
// There is a trade-off between round and communication.
// A larger radix renders a smaller number of rounds but a larger
// communication.
class EqualProtocol {
 public:
  // REQUIRE 1 <= compare_radix <= 8.
  explicit EqualProtocol(const std::shared_ptr<BasicOTProtocols>& base,
                         size_t compare_radix = 4);

  ~EqualProtocol();

  NdArrayRef Compute(const NdArrayRef& inp, size_t bit_width = 0);

  NdArrayRef FlattedCompute(const NdArrayRef& inp, size_t bit_width = 0);

  NdArrayRef BatchCompute(const NdArrayRef& inp, int64_t numel,
                          int64_t bitwidth, int64_t batch_size);

  void set_as_sender(bool flag) { is_sender_ = flag; }

 private:
  NdArrayRef TraversalANDFullBinaryTree(NdArrayRef eq, size_t num_input,
                                        size_t num_digits);

  NdArrayRef TraversalAND(NdArrayRef eq, size_t num_input, size_t num_digits);

  NdArrayRef DoFlattenVersion(const NdArrayRef& inp, size_t bit_width = 0);

  NdArrayRef DoCompute(const NdArrayRef& inp, size_t bit_width = 0);

  NdArrayRef DoBatchCompute(const NdArrayRef& inp, int64_t numelt,
                            int64_t bit_width, int64_t batch_size);

  size_t compare_radix_;
  bool is_sender_{false};
  std::shared_ptr<BasicOTProtocols> basic_ot_prot_;
};

}  // namespace spu::mpc::cheetah
