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
#include <climits>
#include <cstdint>
#include <cstdio>
#include <stack>
#include <vector>

#include "absl/types/span.h"

#include "libspu/core/prelude.h"

namespace psi::mpc {

class BenesNetwork {
 public:
  using gate_t = int8_t;
  constexpr static gate_t kUnknownGate = -1;
  constexpr static gate_t kIdentityGate = 0;
  constexpr static gate_t kSwapGate = 1;
  constexpr static gate_t kEmptyGate = 2;

  explicit BenesNetwork(absl::Span<const int32_t> dest);

  explicit BenesNetwork(uint64_t input_size);

  ~BenesNetwork();

  int64_t total_gates() const { return total_gates_; }

  void get_control_bits(absl::Span<uint8_t> out) const;

  gate_t get_gate_type(int64_t index) const;

  void set_arithmetic_payload(int64_t width) {
    SPU_ENFORCE(width >= 1 and width <= 128);
    is_additive_ = true;
    payload_width_ = width;
  }

  void set_boolean_payload(int64_t width) {
    SPU_ENFORCE(width >= 1 and width <= 128);
    is_additive_ = false;
    payload_width_ = width;
  }

  template <typename T>
  void eval_with_masks(absl::Span<T> src, absl::Span<const T> wire0) const {
    SPU_ENFORCE_EQ((int64_t)src.size(), input_size_);
    SPU_ENFORCE(sizeof(T) * 8 >= (size_t)payload_width_);

    std::function<T(T, T)> add;
    std::function<T(T, T)> sub;
    if (is_additive_) {
      if ((size_t)payload_width_ < sizeof(T) * 8) {
        const T msk = (static_cast<T>(1) << payload_width_) - 1;
        add = [msk](T x, T y) { return (x + y) & msk; };
        sub = [msk](T x, T y) { return (x - y) & msk; };
      } else {
        add = [](T x, T y) { return x + y; };
        sub = [](T x, T y) { return x - y; };
      }
    } else {
      add = [](T x, T y) { return x ^ y; };
      sub = [](T x, T y) { return x ^ y; };
    }

    eval_with_masks<T>(log_aligned_size_, 0, 0, src, wire0, add, sub);
  }

  template <typename T>
  void prepare_correction(absl::Span<T> src, absl::Span<T> corr,
                          absl::Span<const T> ot_msg0,
                          absl::Span<const T> ot_msg1) const {
    SPU_ENFORCE_EQ((int64_t)src.size(), input_size_);
    SPU_ENFORCE(sizeof(T) * 8 >= (size_t)payload_width_);
    std::function<T(T, T)> sub;
    if (is_additive_) {
      if ((size_t)payload_width_ < sizeof(T) * 8) {
        const T msk = (static_cast<T>(1) << payload_width_) - 1;
        sub = [msk](T x, T y) { return (x - y) & msk; };
      } else {
        sub = [](T x, T y) { return x - y; };
      }
    } else {
      sub = [](T x, T y) { return x ^ y; };
    }
    prepare_correction<T>(log_aligned_size_, 0, 0, src, corr, ot_msg0, ot_msg1,
                          sub);
  }

 private:
  template <typename T>
  inline T _addmod(T x, T y) const {
    return x + y;
  }

  template <typename T>
  inline T _xormod(T x, T y) const {
    return x ^ y;
  }

  template <typename T>
  void prepare_correction_small(int64_t n, int64_t lvl_p, int64_t perm_idx,
                                absl::Span<T> src, absl::Span<T> corr,
                                absl::Span<const T> ot_msg0,
                                absl::Span<const T> ot_msg1,
                                std::function<T(T, T)> sub) const {
    const int64_t numel = src.size();
    if (numel < 2) {
      return;
    }

    if (numel == 2) {
      if (n == 1) {
        int64_t idx = lvl_p * (input_size_ / 2) + perm_idx;
        auto w0 = sub(src[0], ot_msg0[idx]);
        auto w1 = sub(src[1], sub(0, ot_msg0[idx]));
        corr[idx] = sub(sub(src[0], w1), ot_msg1[idx]);

        src[0] = w0;
        src[1] = w1;

      } else {
        int64_t idx = (lvl_p + 1) * (input_size_ / 2) + perm_idx;
        auto w0 = sub(src[0], ot_msg0[idx]);
        auto w1 = sub(src[1], sub(0, ot_msg0[idx]));
        corr[idx] = sub(sub(src[0], w1), ot_msg1[idx]);

        src[0] = w0;
        src[1] = w1;
      }
      return;
    }

    // numel = 3
    {
      int64_t idx = lvl_p * (input_size_ / 2) + perm_idx;
      auto w0 = sub(src[0], ot_msg0[idx]);
      auto w1 = sub(src[1], sub(0, ot_msg0[idx]));
      corr[idx] = sub(sub(src[0], w1), ot_msg1[idx]);

      src[0] = w0;
      src[1] = w1;
    }
    {
      int64_t idx = (lvl_p + 1) * (input_size_ / 2) + perm_idx;
      auto w0 = sub(src[1], ot_msg0[idx]);
      auto w1 = sub(src[2], sub(0, ot_msg0[idx]));
      corr[idx] = sub(sub(src[1], w1), ot_msg1[idx]);

      src[1] = w0;
      src[2] = w1;
    }
    {
      int64_t idx = (lvl_p + 2) * (input_size_ / 2) + perm_idx;
      auto w0 = sub(src[0], ot_msg0[idx]);
      auto w1 = sub(src[1], sub(0, ot_msg0[idx]));
      corr[idx] = sub(sub(src[0], w1), ot_msg1[idx]);

      src[0] = w0;
      src[1] = w1;
    }
  }

  template <typename T>
  void prepare_correction(int64_t n, int64_t lvl_p, int64_t perm_idx,
                          absl::Span<T> src, absl::Span<T> corr,
                          absl::Span<const T> ot_msg0,
                          absl::Span<const T> ot_msg1,
                          std::function<T(T, T)> sub) const {
    const int levels = 2 * n - 1;
    const int numel = src.size();
    if (numel <= 3) {
      prepare_correction_small(n, lvl_p, perm_idx, src, corr, ot_msg0, ot_msg1,
                               sub);
      return;
    }

    std::vector<T> bottom1;
    std::vector<T> top1;
    top1.reserve(numel / 2);
    bottom1.reserve(numel / 2);
    for (int64_t i = 0; i < numel - 1; i += 2) {
      int64_t idx = lvl_p * (input_size_ / 2) + perm_idx + i / 2;
      auto w0 = sub(src[i + 0], ot_msg0[idx]);
      auto w1 = sub(src[i + 1], sub(0, ot_msg0[idx]));

      // cr = x0 - w1 - r1
      corr[idx] = sub(sub(src[i + 0], w1), ot_msg1[idx]);

      src[i + 0] = w0;
      src[i + 1] = w1;

      bottom1.push_back(src[i + 0]);
      top1.push_back(src[i + 1]);
    }

    if (numel & 1) {
      top1.push_back(src[numel - 1]);
    }

    prepare_correction(n - 1, lvl_p + 1, perm_idx + numel / 4,
                       absl::MakeSpan(top1), corr, ot_msg0, ot_msg1, sub);

    prepare_correction(n - 1, lvl_p + 1, perm_idx, absl::MakeSpan(bottom1),
                       corr, ot_msg0, ot_msg1, sub);

    for (int64_t i = 0; i < numel - 1; i += 2) {
      int64_t idx = (lvl_p + levels - 1) * (input_size_ / 2) + perm_idx + i / 2;
      auto w0 = sub(bottom1[i / 2], ot_msg0[idx]);
      auto w1 = sub(top1[i / 2], sub(0, ot_msg0[idx]));
      corr[idx] = sub(sub(bottom1[i / 2], w1), ot_msg1[idx]);

      src[i + 0] = w0;
      src[i + 1] = w1;
    }

    if (numel & 1) {
      src[numel - 1] = top1[(numel + 1) / 2 - 1];
    }
  }

  static inline int32_t benes_right_cycle_shift(int32_t num, int32_t logN) {
    return ((num & 1) << (logN - 1)) | (num >> 1);
  }

  inline int64_t compute_gate_index(int64_t level, int64_t perm_idx) const {
    return level * (input_size_ >> 1) + perm_idx;
  }

  template <typename T>
  void eval_with_masks_small(int64_t n, int64_t lvl_p, int64_t perm_idx,
                             absl::Span<T> src, absl::Span<const T> wire0,
                             std::function<T(T, T)> add,
                             std::function<T(T, T)> sub) const {
    const int64_t numel = src.size();
    if (numel <= 1) {
      return;
    }

    if (numel == 2) {
      if (n == 1) {
        auto gidx = compute_gate_index(lvl_p, perm_idx);
        src[0] = add(src[0], wire0[gidx]);
        src[1] = add(src[1], sub(0, wire0[gidx]));
        if (get_gate_type(gidx) == kSwapGate) {
          std::swap(src[0], src[1]);
        }
      } else {
        auto gidx = compute_gate_index(lvl_p + 1, perm_idx);
        src[0] = add(src[0], wire0[gidx]);
        src[1] = add(src[1], sub(0, wire0[gidx]));
        if (get_gate_type(gidx) == kSwapGate) {
          std::swap(src[0], src[1]);
        }
      }
      return;
    }

    // numel = 3
    auto gidx = compute_gate_index(lvl_p, perm_idx);
    src[0] = add(src[0], wire0[gidx]);
    src[1] = add(src[1], sub(0, wire0[gidx]));
    if (get_gate_type(gidx) == kSwapGate) {
      std::swap(src[0], src[1]);
    }

    gidx = compute_gate_index(lvl_p + 1, perm_idx);
    src[1] = add(src[1], wire0[gidx]);
    src[2] = add(src[2], sub(0, wire0[gidx]));
    if (get_gate_type(gidx) == kSwapGate) {
      std::swap(src[1], src[2]);
    }

    gidx = compute_gate_index(lvl_p + 2, perm_idx);
    src[0] = add(src[0], wire0[gidx]);
    src[1] = add(src[1], sub(0, wire0[gidx]));
    if (get_gate_type(gidx) == kSwapGate) {
      std::swap(src[0], src[1]);
    }
  }

  template <typename T>
  void eval_with_masks(int64_t n, int64_t lvl_p, int64_t perm_idx,
                       absl::Span<T> src, absl::Span<const T> wire0,
                       std::function<T(T, T)> add,
                       std::function<T(T, T)> sub) const {
    const int64_t numel = src.size();
    if (numel <= 3) {
      eval_with_masks_small(n, lvl_p, perm_idx, src, wire0, add, sub);
      return;
    }
    const int64_t width = numel / 2;

    std::vector<T> bottom1;
    std::vector<T> top1;
    const int64_t levels = 2 * n - 1;

    auto base_idx = compute_gate_index(lvl_p, perm_idx);
    for (int64_t i = 0; i < numel - 1; i += 2) {
      auto gidx = base_idx + i / 2;
      auto s = get_gate_type(gidx);

      src[i + 0] = add(src[i + 0], wire0[gidx]);
      src[i + 1] = add(src[i + 1], sub(0, wire0[gidx]));

      if (benes_right_cycle_shift((i + 0) ^ s, n) < width) {
        bottom1.push_back(src[i + 0]);
      } else {
        top1.push_back(src[i + 0]);
      }

      if (benes_right_cycle_shift((i + 1) ^ s, n) < width) {
        bottom1.push_back(src[i + 1]);
      } else {
        top1.push_back(src[i + 1]);
      }
    }

    if (numel & 1) {
      top1.push_back(src[numel - 1]);
    }

    eval_with_masks(n - 1, lvl_p + 1, perm_idx + numel / 4,
                    absl::MakeSpan(top1), wire0, add, sub);
    eval_with_masks(n - 1, lvl_p + 1, perm_idx, absl::MakeSpan(bottom1), wire0,
                    add, sub);

    base_idx = compute_gate_index(lvl_p + levels - 1, perm_idx);
    for (int64_t i = 0; i < numel - 1; i += 2) {
      auto gidx = base_idx + i / 2;
      auto s = get_gate_type(gidx);
      int x;
      if ((x = benes_right_cycle_shift((i + 0) ^ s, n)) < width) {
        src[i + 0] = bottom1[x];
      } else {
        src[i + 0] = top1[i / 2];
      }

      if ((x = benes_right_cycle_shift((i + 1) ^ s, n)) < width) {
        src[i + 1] = bottom1[x];
      } else {
        src[i + 1] = top1[i / 2];
      }

      auto w0 = wire0[gidx];
      auto w1 = sub(0, w0);

      src[i + 0] = add(src[i + 0], s ? w1 : w0);
      src[i + 1] = add(src[i + 1], s ? w0 : w1);
    }

    if (numel & 1) {
      src[numel - 1] = top1[(numel + 1) / 2 - 1];
    }
  }

  void benes_depth_first_search(int32_t idx, int8_t route);

  void gen_benes_route(int32_t subLogN, int32_t lvl_p, int32_t perm_idx,
                       absl::Span<const int32_t> src,
                       absl::Span<const int32_t> dest);

  void gen_benes_route_small(int32_t subLogN, int32_t lvl_p, int32_t perm_idx,
                             absl::Span<const int32_t> src,
                             absl::Span<const int32_t> dest);

  void set_gate_type(int64_t level, int64_t perm_idx, gate_t type);

  void set_gate_type(int64_t index, gate_t type);

  gate_t get_gate_type(int64_t level, int64_t perm_idx) const;

  int64_t input_size_;        // n
  int64_t log_aligned_size_;  // ceil(log2(n))
  int64_t total_levels_;      // 2 * log_aligned_size_ - 1
  int64_t total_gates_;       // (n / 2) * total_levels_

  bool is_additive_ = true;
  int64_t payload_width_ = 0;

  std::vector<int32_t> benes_perm;
  std::vector<int32_t> benes_inv_perm;
  std::vector<int8_t> benes_path;
  std::vector<gate_t> gates_;
};

}  // namespace psi::mpc
