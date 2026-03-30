#include <cstdint>
#include <stack>
#include <vector>

#include "absl/types/span.h"
#include "yacl/base/int128.h"
#ifndef MPC4J_NATIVE_TOOL_BENES_NETWORK_HPP
#define MPC4J_NATIVE_TOOL_BENES_NETWORK_HPP

class BenesNetwork {
 public:
  using scalar_t = uint128_t;
  int64_t width_;
  explicit BenesNetwork(std::vector<int32_t> &dest) {
    auto n = (int32_t)dest.size();
    auto logN = int32_t(ceil(log2(n)));
    int32_t levels = 2 * logN - 1;
    benes_perm.resize(n);
    benes_inv_perm.resize(n);
    benes_network.resize(levels);
    for (int32_t i = 0; i < levels; ++i) {
      benes_network[i].resize(n / 2);
      std::fill(benes_network[i].begin(), benes_network[i].end(), -1);
    }
    // the input is [0, 1, ..., n)
    std::vector<int32_t> src(n);
    for (int32_t i = 0; i < n; ++i) {
      src[i] = i;
    }
    width_ = n / 2;
    gen_benes_route(logN, 0, 0, src, dest);
  }

  std::vector<std::vector<int8_t>> get_benes_network() const {
    return benes_network;
  }

  std::vector<uint8_t> get_control_bits() const {
    std::vector<uint8_t> out(benes_network.size() * benes_network[0].size());
    for (size_t i = 0; i < benes_network.size(); ++i) {
      for (size_t j = 0; j < benes_network[i].size(); ++j) {
        out[i * benes_network[i].size() + j] = benes_network[i][j] > 0;
      }
    }
    return out;
  }

  void eval(std::vector<scalar_t> &src) const {
    int N = src.size();
    int n = std::ceil(std::log2(N));
    eval_benes(n, 0, 0, src);
  }

  void eval_with_wire_masks(absl::Span<scalar_t> src,
                            absl::Span<const scalar_t> wire0,
                            absl::Span<const scalar_t> wire1) const {
    int N = src.size();
    int n = std::ceil(std::log2(N));
    eval_with_wire_masks(n, 0, 0, src, wire0, wire1);
  }

  void eval_with_wire_masks_additive(absl::Span<scalar_t> src,
                                     absl::Span<const scalar_t> wire0,
                                     absl::Span<const scalar_t> wire1) const {
    int N = src.size();
    int n = std::ceil(std::log2(N));
    eval_with_wire_masks_additive(n, 0, 0, src, wire0, wire1);
  }

  ~BenesNetwork() {
    benes_perm.clear();
    benes_perm.shrink_to_fit();
    benes_inv_perm.clear();
    benes_perm.shrink_to_fit();
    benes_network.clear();
    benes_network.shrink_to_fit();
    benes_path.clear();
    benes_path.shrink_to_fit();
  }

  void prepare_correction(
      absl::Span<scalar_t> src, absl::Span<std::array<scalar_t, 2>> corrections,
      absl::Span<const std::array<scalar_t, 2>> ot_msg,
      absl::Span<const std::array<scalar_t, 2>> prp_ot_msg) const {
    int64_t n = src.size();
    int64_t ln = std::ceil(std::log2(n));
    prepare_correction(n, ln, 0, 0, src, corrections, ot_msg, prp_ot_msg);
  }

  void prepare_correction_additive(
      absl::Span<scalar_t> src, absl::Span<std::array<scalar_t, 2>> corrections,
      absl::Span<const std::array<scalar_t, 2>> ot_msg,
      absl::Span<const std::array<scalar_t, 2>> prp_ot_msg) const {
    int64_t n = src.size();
    int64_t ln = std::ceil(std::log2(n));
    prepare_correction_additive(n, ln, 0, 0, src, corrections, ot_msg,
                                prp_ot_msg);
  }

 private:
  void eval_benes(int n, int lvl_p, int perm_idx,
                  std::vector<scalar_t> &src) const;

  void eval_with_wire_masks(int64_t n, int64_t lvl_p, int64_t perm_idx,
                            std::vector<scalar_t> &src,
                            const std::vector<scalar_t> &wire0,
                            const std::vector<scalar_t> &wire1) const;

  void eval_with_wire_masks(int64_t n, int64_t lvl_p, int64_t perm_idx,
                            absl::Span<scalar_t> src,
                            absl::Span<const scalar_t> wire0,
                            absl::Span<const scalar_t> wire1) const;

  void prepare_correction(
      int64_t total_numel, int64_t n, int64_t lvl_p, int64_t perm_idx,
      absl::Span<scalar_t> src, absl::Span<std::array<scalar_t, 2>> corrections,
      absl::Span<const std::array<scalar_t, 2>> ot_msg,
      absl::Span<const std::array<scalar_t, 2>> prp_ot_msg) const;

  void prepare_correction_additive(
      int64_t total_numel, int64_t n, int64_t lvl_p, int64_t perm_idx,
      absl::Span<scalar_t> src, absl::Span<std::array<scalar_t, 2>> corrections,
      absl::Span<const std::array<scalar_t, 2>> ot_msg,
      absl::Span<const std::array<scalar_t, 2>> prp_ot_msg) const;

  void eval_with_wire_masks_additive(int64_t n, int64_t lvl_p, int64_t perm_idx,
                                     absl::Span<scalar_t> src,
                                     absl::Span<const scalar_t> wire0,
                                     absl::Span<const scalar_t> wire1) const;

  /**
   * [N] -> [T]
   */
  std::vector<int32_t> benes_perm;
  /**
   * [N] <- [T]
   */
  std::vector<int32_t> benes_inv_perm;
  /**
   * benes_network
   */
  std::vector<std::vector<int8_t>> benes_network;
  /**
   * path
   */
  std::vector<int8_t> benes_path;

  static int32_t benes_right_cycle_shift(int32_t num, int32_t logN) {
    return ((num & 1) << (logN - 1)) | (num >> 1);
  }

  /**
   * depth-first search.
   */
  void benes_depth_first_search(int32_t idx, int8_t route) {
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
        pathStack.push(
            {idxRoutePair.first ^ 1, idxRoutePair.second ^ (int8_t)1});
      }
      idx = benes_perm[benes_inv_perm[idxRoutePair.first] ^ 1];
      if (benes_path[idx] < 0) {
        pathStack.push({idx, idxRoutePair.second ^ (int8_t)1});
      }
    }
  }

  void gen_benes_route(int32_t subLogN, int32_t lvl_p, int32_t perm_idx,
                       const std::vector<int32_t> &src,
                       const std::vector<int32_t> &dest) {
    auto subN = (int32_t)src.size();
    if (subN == 2) {
      if (subLogN == 1) {
        // logN == 1, we have 2 * log(N) - 1 = 1 level (█)
        benes_network[lvl_p][perm_idx] = (int8_t)(src[0] != dest[0]);
      } else {
        // logN == 2，we have 2 * logN - 1 = 3 levels (□ █ □).
        benes_network[lvl_p][perm_idx] = 2;
        benes_network[lvl_p + 1][perm_idx] = (int8_t)(src[0] != dest[0]);
        benes_network[lvl_p + 2][perm_idx] = 2;
      }
    } else if (subN == 3) {
      if (src[0] == dest[0]) {
        /*
         * 0 -> 0，1 -> 1，2 -> 2, the network is:
         * █ □ █ = 0   0
         * □ █ □     0
         *
         * 0 -> 0，1 -> 2，2 -> 1, the network is:
         * █ □ █ = 0   0
         * □ █ □     1
         */
        benes_network[lvl_p][perm_idx] = (int8_t)0;
        benes_network[lvl_p + 2][perm_idx] = (int8_t)0;
        if (src[1] == dest[1]) {
          benes_network[lvl_p + 1][perm_idx] = (int8_t)0;
        } else {
          benes_network[lvl_p + 1][perm_idx] = (int8_t)1;
        }
      } else if (src[0] == dest[1]) {
        /*
         * 0 -> 1，1 -> 0，2 -> 2, the network is:
         * █ □ █ = 0   1
         * □ █ □     0
         *
         * 0 -> 1，1 -> 2，2 -> 0, the network is:
         * █ □ █ = 0   1
         * □ █ □     1
         */
        benes_network[lvl_p][perm_idx] = (int8_t)0;
        benes_network[lvl_p + 2][perm_idx] = (int8_t)1;
        if (src[1] == dest[0]) {
          benes_network[lvl_p + 1][perm_idx] = (int8_t)0;
        } else {
          benes_network[lvl_p + 1][perm_idx] = (int8_t)1;
        }
      } else {
        /*
         * 0 -> 2，1 -> 0，2 -> 1, the network is:
         * █ □ █ = 1   0
         * □ █ □     1
         *
         * 0 -> 2，1 -> 1，2 -> 0, the network is:
         * █ □ █ = 1   1
         * □ █ □     1
         */
        benes_network[lvl_p][perm_idx] = (int8_t)1;
        benes_network[lvl_p + 1][perm_idx] = (int8_t)1;
        if (src[1] == dest[0]) {
          benes_network[lvl_p + 2][perm_idx] = (int8_t)0;
        } else {
          benes_network[lvl_p + 2][perm_idx] = (int8_t)1;
        }
      }
      return;
    } else {
      int32_t i, j, x;
      uint8_t s;
      int32_t subLevel = 2 * subLogN - 1;
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
      for (i = 0; i < subN; ++i) {
        benes_inv_perm[src[i]] = i;
      }
      for (i = 0; i < subN; ++i) {
        benes_perm[i] = benes_inv_perm[dest[i]];
      }
      for (i = 0; i < subN; ++i) {
        benes_inv_perm[benes_perm[i]] = i;
      }
      // shorten the array
      benes_path.resize(subN);
      // path, initialized by -1, we use 2 for empty node
      std::fill(benes_path.begin(), benes_path.end(), (int8_t)-1);
      // handling odd n
      if (subN % 2 == 1) {
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
      for (i = 0; i < subN; ++i) {
        if (benes_path[i] < 0) {
          benes_depth_first_search(i, (int8_t)0);
        }
      }
      // create left part of the network.
      for (i = 0; i < subN - 1; i += 2) {
        benes_network[lvl_p][perm_idx + i / 2] = benes_path[i];
        for (j = 0; j < 2; ++j) {
          x = benes_right_cycle_shift((i | j) ^ benes_path[i], subLogN);
          if (x < subN / 2) {
            topSrc.push_back(src[i | j]);
          } else {
            bottomSrc.push_back(src[i | j]);
          }
        }
      }
      if (subN % 2 == 1) {
        // add one more switch for the odd case.
        bottomSrc.push_back(src[subN - 1]);
      }
      // create right part of the subnetwork.
      for (i = 0; i < subN - 1; i += 2) {
        s = benes_network[lvl_p + subLevel - 1][perm_idx + i / 2] =
            benes_path[benes_perm[i]];
        for (j = 0; j < 2; ++j) {
          x = benes_right_cycle_shift((i | j) ^ s, subLogN);
          if (x < subN / 2) {
            topDest[i / 2] = src[benes_perm[i | j]];
          } else {
            bottomDest[i / 2] = src[benes_perm[i | j]];
          }
        }
      }
      if (subN % 2 == 1) {
        // add one more switch for the odd case.
        bottomDest[subN / 2] = dest[subN - 1];
      }
      // create top subnetwork, with (log(N) - 1) levels
      gen_benes_route(subLogN - 1, lvl_p + 1, perm_idx, topSrc, topDest);
      // create bottom subnetwork with (log(N) - 1) levels.
      gen_benes_route(subLogN - 1, lvl_p + 1, perm_idx + subN / 4, bottomSrc,
                      bottomDest);
    }
  }
};

#endif  // MPC4J_NATIVE_TOOL_BENES_NETWORK_HPP
