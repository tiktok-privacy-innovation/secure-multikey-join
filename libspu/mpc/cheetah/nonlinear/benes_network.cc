#include "libspu/mpc/cheetah/nonlinear/benes_network.h"

#include <algorithm>
#include <numeric>
#include <random>

#include "libspu/core/prelude.h"

namespace spu::mpc::cheetah {

BenesNetwork::BenesNetwork(absl::Span<const int32_t> dest)
    : input_size_(dest.size()),
      log_aligned_size_(absl::bit_width(absl::bit_ceil(dest.size())) - 1),
      total_levels_(2 * log_aligned_size_ - 1),
      total_gates_(total_levels_ * (input_size_ / 2)) {
  benes_perm.resize(input_size_);
  benes_inv_perm.resize(input_size_);
  gates_.resize(total_gates_, kUnknownGate);
  // src is [0, 1, ..., n)
  std::vector<int32_t> src(input_size_);
  std::iota(src.begin(), src.end(), 0);
  gen_benes_route(log_aligned_size_, 0, 0, src, dest);
}

BenesNetwork::BenesNetwork(uint64_t input_size)
    : input_size_((int64_t)input_size),
      log_aligned_size_(absl::bit_width(absl::bit_ceil(input_size)) - 1),
      total_levels_(2 * log_aligned_size_ - 1),
      total_gates_(total_levels_ * (input_size_ / 2)) {
  gates_.resize(total_gates_, kUnknownGate);
  std::uniform_int_distribution<gate_t> r3(0, 1);
  std::default_random_engine re(std::random_device{}());
  std::generate_n(gates_.begin(), gates_.size(), [&]() { return r3(re); });
}

void BenesNetwork::get_control_bits(absl::Span<uint8_t> out) const {
  SPU_ENFORCE_EQ((int64_t)out.size(), total_gates_);
  std::transform(gates_.begin(), gates_.end(), out.data(),
                 [](gate_t g) -> uint8_t { return g == kSwapGate; });
}

BenesNetwork::~BenesNetwork() {
  benes_perm.clear();
  benes_perm.shrink_to_fit();
  benes_inv_perm.clear();
  benes_perm.shrink_to_fit();
  benes_path.clear();
  benes_path.shrink_to_fit();
  gates_.clear();
  gates_.shrink_to_fit();
}

void BenesNetwork::benes_depth_first_search(int32_t idx, int8_t route) {
  std::stack<std::pair<int32_t, int8_t>> pathStack;
  pathStack.push({idx, route});
  std::pair<int32_t, int8_t> idxRoutePair;
  while (!pathStack.empty()) {
    idxRoutePair = pathStack.top();
    pathStack.pop();
    benes_path[idxRoutePair.first] = idxRoutePair.second;
    // if the next item in the vertical array is unassigned
    if (benes_path[idxRoutePair.first ^ 1] < 0) {
      // the next item is always assigned the opposite of this item,
      // unless it was part of path/cycle of previous node
      pathStack.push({idxRoutePair.first ^ 1, idxRoutePair.second ^ (int8_t)1});
    }
    idx = benes_perm[benes_inv_perm[idxRoutePair.first] ^ 1];
    if (benes_path[idx] < 0) {
      pathStack.push({idx, idxRoutePair.second ^ (int8_t)1});
    }
  }
}

void BenesNetwork::set_gate_type(int64_t level, int64_t perm_idx, gate_t type) {
  SPU_ENFORCE(level >= 0 && level < total_levels_);
  SPU_ENFORCE(perm_idx >= 0 && perm_idx < input_size_ / 2);
  set_gate_type(compute_gate_index(level, perm_idx), type);
}

void BenesNetwork::set_gate_type(int64_t index, gate_t type) {
  SPU_ENFORCE(index >= 0 and index < total_gates_);
  gates_[index] = type;
}

BenesNetwork::gate_t BenesNetwork::get_gate_type(int64_t level,
                                                 int64_t perm_idx) const {
  SPU_ENFORCE(level >= 0 && level < total_levels_);
  SPU_ENFORCE(perm_idx >= 0 && perm_idx < input_size_ / 2);
  return get_gate_type(compute_gate_index(level, perm_idx));
}

BenesNetwork::gate_t BenesNetwork::get_gate_type(int64_t index) const {
  SPU_ENFORCE(index >= 0 and index < total_gates_);
  return gates_[index];
}

void BenesNetwork::gen_benes_route_small(int32_t subLogN, int32_t lvl_p,
                                         int32_t perm_idx,
                                         absl::Span<const int32_t> src,
                                         absl::Span<const int32_t> dest) {
  auto subN = (int32_t)src.size();
  SPU_ENFORCE(subN >= 0 && subN <= 3, "src.size {}", src.size());
  if (subN <= 1) {
    return;
  }

  if (subN == 2) {
    if (subLogN == 1) {
      set_gate_type(lvl_p, perm_idx,
                    src[0] != dest[0] ? kSwapGate : kIdentityGate);
    } else {
      set_gate_type(lvl_p + 0, perm_idx, kEmptyGate);
      set_gate_type(lvl_p + 1, perm_idx,
                    src[0] != dest[0] ? kSwapGate : kIdentityGate);
      set_gate_type(lvl_p + 2, perm_idx, kEmptyGate);
    }
    return;
  }

  // subN = 3
  if (src[0] == dest[0]) {
    set_gate_type(lvl_p + 0, perm_idx, kIdentityGate);
    set_gate_type(lvl_p + 1, perm_idx,
                  src[1] != dest[1] ? kSwapGate : kIdentityGate);
    set_gate_type(lvl_p + 2, perm_idx, kIdentityGate);
  } else if (src[0] == dest[1]) {
    set_gate_type(lvl_p + 0, perm_idx, kIdentityGate);
    set_gate_type(lvl_p + 1, perm_idx,
                  src[1] != dest[0] ? kSwapGate : kIdentityGate);
    set_gate_type(lvl_p + 2, perm_idx, kSwapGate);
  } else {
    set_gate_type(lvl_p + 0, perm_idx, kSwapGate);
    set_gate_type(lvl_p + 1, perm_idx, kSwapGate);
    set_gate_type(lvl_p + 2, perm_idx,
                  src[1] != dest[0] ? kSwapGate : kIdentityGate);
  }
}

void BenesNetwork::gen_benes_route(int32_t subLogN, int32_t lvl_p,
                                   int32_t perm_idx,
                                   absl::Span<const int32_t> src,
                                   absl::Span<const int32_t> dest) {
  auto subN = (int32_t)src.size();
  if (subN <= 3) {
    gen_benes_route_small(subLogN, lvl_p, perm_idx, src, dest);
    return;
  }

  const int32_t subLevel = 2 * subLogN - 1;
  // top subnetwork map, with size Math.floor(n / 2)
  std::vector<int32_t> topSrc(0);
  std::vector<int32_t> topDest(subN / 2);
  // bottom subnetwork map, with size Math.ceil(n / 2)
  std::vector<int32_t> bottomSrc(0);
  std::vector<int32_t> bottomDest(int(ceil(subN * 0.5)));
  // create forward/backward lookup tables
  // subSrcList stores the position map. For example, src = [2, 4, 6],
  // dest = [6, 4, 2]. We re-organize the map to the form [0, subN - 1) ->
  // [0, subN - 1)
  for (int32_t i = 0; i < subN; ++i) {
    benes_inv_perm[src[i]] = i;
  }
  for (int32_t i = 0; i < subN; ++i) {
    benes_perm[i] = benes_inv_perm[dest[i]];
  }
  for (int32_t i = 0; i < subN; ++i) {
    benes_inv_perm[benes_perm[i]] = i;
  }
  // shorten the array
  benes_path.resize(subN);
  // path, initialized by -1, we use 2 for empty node
  std::fill(benes_path.begin(), benes_path.end(), (int8_t)-1);
  // handling odd n
  if (subN & 1) {
    // the last node directly links to the bottom subnetwork.
    benes_path[subN - 1] = (int8_t)1;
    benes_path[benes_perm[subN - 1]] = (int8_t)1;
    // if values - 1 == benes_perm[values - 1], then the last one is also
    // a direct link. Handle other cases.
    if (benes_perm[subN - 1] != subN - 1) {
      int32_t idx = benes_perm[benes_inv_perm[subN - 1] ^ 1];
      benes_depth_first_search(idx, (int8_t)0);
    }
  }
  // set other switches
  for (int32_t i = 0; i < subN; ++i) {
    if (benes_path[i] < 0) {
      benes_depth_first_search(i, (int8_t)0);
    }
  }
  // create left part of the network.
  int64_t gidx_base = compute_gate_index(lvl_p, perm_idx);
  for (int32_t i = 0; i < subN - 1; i += 2) {
    set_gate_type(gidx_base + i / 2,
                  benes_path[i] == 0 ? kIdentityGate : kSwapGate);
    for (int32_t j = 0; j < 2; ++j) {
      auto x = benes_right_cycle_shift((i | j) ^ benes_path[i], subLogN);
      if (x < subN / 2) {
        topSrc.push_back(src[i | j]);
      } else {
        bottomSrc.push_back(src[i | j]);
      }
    }
  }
  if (subN & 1) {
    // add one more switch for the odd case.
    bottomSrc.push_back(src[subN - 1]);
  }
  // create right part of the subnetwork.
  gidx_base = compute_gate_index(lvl_p + subLevel - 1, perm_idx);
  for (int32_t i = 0; i < subN - 1; i += 2) {
    auto s = benes_path[benes_perm[i]];
    set_gate_type(gidx_base + i / 2, s == 0 ? kIdentityGate : kSwapGate);
    for (int32_t j = 0; j < 2; ++j) {
      auto x = benes_right_cycle_shift((i | j) ^ s, subLogN);
      if (x < subN / 2) {
        topDest[i / 2] = src[benes_perm[i | j]];
      } else {
        bottomDest[i / 2] = src[benes_perm[i | j]];
      }
    }
  }
  if (subN & 1) {
    // add one more switch for the odd case.
    bottomDest[subN / 2] = dest[subN - 1];
  }
  // create top subnetwork, with (log(N) - 1) levels
  gen_benes_route(subLogN - 1, lvl_p + 1, perm_idx, absl::MakeSpan(topSrc),
                  absl::MakeSpan(topDest));
  // create bottom subnetwork with (log(N) - 1) levels.
  gen_benes_route(subLogN - 1, lvl_p + 1, perm_idx + subN / 4,
                  absl::MakeSpan(bottomSrc), absl::MakeSpan(bottomDest));
}

}  // namespace spu::mpc::cheetah
