#include "libspu/mpc/cheetah/bn.h"

void BenesNetwork::eval_benes(int n, int lvl_p, int perm_idx,
                              std::vector<scalar_t>& src) const {
  int levels, i, j, x, s std::vector<scalar_t> bottom1;
  std::vector<scalar_t> top1;
  int values = src.size();
  scalar_t temp;

  if (values == 2) {
    if (n == 1) {
      if (benes_network[lvl_p][perm_idx] == 1) {
        temp = src[0];
        src[0] = src[1];
        src[1] = temp;
      }
    } else if (benes_network[lvl_p + 1][perm_idx] == 1) {
      temp = src[0];
      src[0] = src[1];
      src[1] = temp;
    }
    return;
  }

  if (values == 3) {
    if (benes_network[lvl_p][perm_idx] == 1) {
      temp = src[0];
      src[0] = src[1];
      src[1] = temp;
    }
    if (benes_network[lvl_p + 1][perm_idx] == 1) {
      temp = src[1];
      src[1] = src[2];
      src[2] = temp;
    }
    if (benes_network[lvl_p + 2][perm_idx] == 1) {
      temp = src[0];
      src[0] = src[1];
      src[1] = temp;
    }
    return;
  }

  levels = 2 * n - 1;

  for (i = 0; i < values - 1; i += 2) {
    int s = benes_network[lvl_p][perm_idx + i / 2];
    for (j = 0; j < 2; ++j) {
      x = benes_right_cycle_shift((i | j) ^ s, n);
      if (x < values / 2)
        bottom1.push_back(src[i | j]);
      else
        top1.push_back(src[i | j]);
    }
  }
  if (values % 2 == 1) {
    top1.push_back(src[values - 1]);
  }

  eval_benes(n - 1, lvl_p + 1, perm_idx, bottom1);
  eval_benes(n - 1, lvl_p + 1, perm_idx + values / 4, top1);

  for (i = 0; i < values - 1; i += 2) {
    s = benes_network[lvl_p + levels - 1][perm_idx + i / 2];
    for (j = 0; j < 2; ++j) {
      x = benes_right_cycle_shift((i | j) ^ s, n);
      if (x < values / 2)
        src[i | j] = bottom1[x];
      else {
        src[i | j] = top1[i / 2];
      }
    }
  }

  int idx = int(ceil(values * 0.5));
  if (values % 2 == 1) {
    src[values - 1] = top1[idx - 1];
  }
}

void BenesNetwork::eval_with_wire_masks_additive(
    int64_t n, int64_t lvl_p, int64_t perm_idx, absl::Span<scalar_t> src,
    absl::Span<const scalar_t> wire0, absl::Span<const scalar_t> wire1) const {
  const int64_t numel = src.size();
  const int64_t width = numel / 2;
  const scalar_t mod_mask = static_cast<scalar_t>(-1);

  if (numel == 2) {
    if (n == 1) {
      src[0] += wire0[lvl_p * width_ + perm_idx];
      src[1] += wire1[lvl_p * width_ + perm_idx];
      if (benes_network[lvl_p][perm_idx] == 1) {
        std::swap(src[0], src[1]);
      }
    } else {
      src[0] += wire0[(lvl_p + 1) * width_ + perm_idx];
      src[1] += wire1[(lvl_p + 1) * width_ + perm_idx];
      if (benes_network[lvl_p + 1][perm_idx] == 1) {
        std::swap(src[0], src[1]);
      }
    }
  } else if (numel == 3) {
    src[0] += wire0[lvl_p * width_ + perm_idx];
    src[1] += wire1[lvl_p * width_ + perm_idx];
    if (benes_network[lvl_p][perm_idx] == 1) {
      std::swap(src[0], src[1]);
    }

    src[1] += wire0[(lvl_p + 1) * width_ + perm_idx];
    src[2] += wire1[(lvl_p + 1) * width_ + perm_idx];
    if (benes_network[lvl_p + 1][perm_idx] == 1) {
      std::swap(src[1], src[2]);
    }

    src[0] += wire0[(lvl_p + 2) * width_ + perm_idx];
    src[1] += wire1[(lvl_p + 2) * width_ + perm_idx];
    if (benes_network[lvl_p + 2][perm_idx] == 1) {
      std::swap(src[0], src[1]);
    }
  }

  if (numel <= 3) {
    for (auto& u : src) {
      u &= mod_mask;
    }
    return;
  }

  std::vector<scalar_t> bottom1;
  std::vector<scalar_t> top1;
  const int64_t levels = 2 * n - 1;

  for (int64_t i = 0; i < numel - 1; i += 2) {
    auto s = benes_network[lvl_p][perm_idx + i / 2];

    src[i + 0] += wire0[lvl_p * width_ + perm_idx + i / 2];
    src[i + 1] += wire1[lvl_p * width_ + perm_idx + i / 2];
    src[i + 0] &= mod_mask;
    src[i + 1] &= mod_mask;

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

  eval_with_wire_masks_additive(n - 1, lvl_p + 1, perm_idx,
                                absl::MakeSpan(bottom1), wire0, wire1);
  eval_with_wire_masks_additive(n - 1, lvl_p + 1, perm_idx + numel / 4,
                                absl::MakeSpan(top1), wire0, wire1);

  for (int64_t i = 0; i < numel - 1; i += 2) {
    auto s = benes_network[lvl_p + levels - 1][perm_idx + i / 2];
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

    auto w0 = wire0[(lvl_p + levels - 1) * width_ + perm_idx + i / 2];
    auto w1 = wire1[(lvl_p + levels - 1) * width_ + perm_idx + i / 2];

    src[i + 0] += (s ? w1 : w0);
    src[i + 1] += (s ? w0 : w1);

    src[i + 0] &= mod_mask;
    src[i + 1] &= mod_mask;
  }

  if (numel & 1) {
    int idx = int(std::ceil(numel * 0.5));
    src[numel - 1] = top1[idx - 1];
  }
}

void BenesNetwork::eval_with_wire_masks(
    int64_t n, int64_t lvl_p, int64_t perm_idx, std::vector<scalar_t>& src,
    const std::vector<scalar_t>& wire0,
    const std::vector<scalar_t>& wire1) const {
  const int64_t numel = src.size();
  const int64_t width = numel / 2;

  if (numel == 2) {
    if (n == 1) {
      src[0] ^= wire0[lvl_p * width_ + perm_idx];
      src[1] ^= wire1[lvl_p * width_ + perm_idx];
      if (benes_network[lvl_p][perm_idx] == 1) {
        std::swap(src[0], src[1]);
      }
    } else {
      src[0] ^= wire0[(lvl_p + 1) * width_ + perm_idx];
      src[1] ^= wire1[(lvl_p + 1) * width_ + perm_idx];
      if (benes_network[lvl_p + 1][perm_idx] == 1) {
        std::swap(src[0], src[1]);
      }
    }
    return;
  }

  if (numel == 3) {
    src[0] ^= wire0[lvl_p * width_ + perm_idx];
    src[1] ^= wire1[lvl_p * width_ + perm_idx];
    if (benes_network[lvl_p][perm_idx] == 1) {
      std::swap(src[0], src[1]);
    }

    src[1] ^= wire0[(lvl_p + 1) * width_ + perm_idx];
    src[2] ^= wire1[(lvl_p + 1) * width_ + perm_idx];
    if (benes_network[lvl_p + 1][perm_idx] == 1) {
      std::swap(src[1], src[2]);
    }

    src[0] ^= wire0[(lvl_p + 2) * width_ + perm_idx];
    src[1] ^= wire1[(lvl_p + 2) * width_ + perm_idx];
    if (benes_network[lvl_p + 2][perm_idx] == 1) {
      std::swap(src[0], src[1]);
    }
    return;
  }

  std::vector<scalar_t> bottom1;
  std::vector<scalar_t> top1;
  const int64_t levels = 2 * n - 1;

  for (int64_t i = 0; i < numel - 1; i += 2) {
    auto s = benes_network[lvl_p][perm_idx + i / 2];

    src[i + 0] = src[i + 0] ^ wire0[lvl_p * width_ + perm_idx + i / 2];
    src[i + 1] = src[i + 1] ^ wire1[lvl_p * width_ + perm_idx + i / 2];

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

  eval_with_wire_masks(n - 1, lvl_p + 1, perm_idx, bottom1, wire0, wire1);
  eval_with_wire_masks(n - 1, lvl_p + 1, perm_idx + numel / 4, top1, wire0,
                       wire1);

  for (int64_t i = 0; i < numel - 1; i += 2) {
    auto s = benes_network[lvl_p + levels - 1][perm_idx + i / 2];
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

    auto w0 = wire0[(lvl_p + levels - 1) * width_ + perm_idx + i / 2];
    auto w1 = wire1[(lvl_p + levels - 1) * width_ + perm_idx + i / 2];

    src[i] ^= (s ? w1 : w0);
    src[i + 1] ^= (s ? w0 : w1);
  }

  if (numel & 1) {
    int idx = int(std::ceil(numel * 0.5));
    src[numel - 1] = top1[idx - 1];
  }
}

void BenesNetwork::prepare_correction_additive(
    int64_t total_numel, int64_t n, int64_t lvl_p, int64_t perm_idx,
    absl::Span<scalar_t> src, absl::Span<std::array<scalar_t, 2>> corrections,
    absl::Span<const std::array<scalar_t, 2>> ot_msg,
    absl::Span<const std::array<scalar_t, 2>> prp_ot_msg) const {
  // ot message M0 = m0 + w0 || m1 + w1
  //            M1 = m0 + w1 || m1 + w0
  // for each switch:
  // - top wire m0 w0 (input/output)
  // - bottom wires m1, w1
  const int levels = 2 * n - 1;
  const int values = src.size();
  std::vector<scalar_t> bottom1;
  std::vector<scalar_t> top1;
  const scalar_t mod_mask = static_cast<scalar_t>(-1);

  if (values == 2) {
    if (n == 1) {
      // rot message = (r0, r1)
      // expand
      //   M0 = r0 | H(r0)
      //   M1 = r1 | H(r1)
      // parse
      //   M0 = -(w0 + m0) | -(w1 + m1)
      //   M1 = -(w0 + m1) | -(w1 + m0)
      //
      // share on P1 are always
      //   w0 = m0 - r0
      //   w1 = m1 - H(r0)
      // share on P2 (bit = 0)
      //   x0 - m0 + r0 + w0 = x0 => w0 = m0 - r0
      //   x1 - m1 + H(r0)) + w1 = x1 => w1 = m1 - H(r0)
      // share on P2 (bit = 1)
      // Find cr0, cr1 such that
      //   x0 - m0 + r1 + cr0 + w1 = x0
      //   x1 - m1 + H(r1)) + cr1 + w0 = x1
      //   cr0 = m0 - r1 - w1
      //   cr1 = m1 - H(r1) - w0
      int64_t idx = lvl_p * (total_numel / 2) + perm_idx;
      auto w0 = (src[0] - ot_msg[idx][0]) & mod_mask;
      auto w1 = (src[1] - prp_ot_msg[idx][0]) & mod_mask;
      corrections[idx][0] = (src[0] - w1 - ot_msg[idx][1]) & mod_mask;
      corrections[idx][1] = (src[1] - w0 - prp_ot_msg[idx][1]) & mod_mask;

      src[0] = w0;
      src[1] = w1;
    } else {
      int64_t idx = (lvl_p + 1) * (total_numel / 2) + perm_idx;
      auto w0 = (src[0] - ot_msg[idx][0]) & mod_mask;
      auto w1 = (src[1] - prp_ot_msg[idx][0]) & mod_mask;
      corrections[idx][0] = (src[0] - w1 - ot_msg[idx][1]) & mod_mask;
      corrections[idx][1] = (src[1] - w0 - prp_ot_msg[idx][1]) & mod_mask;

      src[0] = w0;
      src[1] = w1;
    }
    return;
  }

  if (values == 3) {
    {
      int64_t idx = lvl_p * (total_numel / 2) + perm_idx;
      auto w0 = (src[0] - ot_msg[idx][0]) & mod_mask;
      auto w1 = (src[1] - prp_ot_msg[idx][0]) & mod_mask;
      corrections[idx][0] = (src[0] - w1 - ot_msg[idx][1]) & mod_mask;
      corrections[idx][1] = (src[1] - w0 - prp_ot_msg[idx][1]) & mod_mask;

      src[0] = w0;
      src[1] = w1;
    }
    {
      int64_t idx = (lvl_p + 1) * (total_numel / 2) + perm_idx;
      auto w0 = (src[1] - ot_msg[idx][0]) & mod_mask;
      auto w1 = (src[2] - prp_ot_msg[idx][0]) & mod_mask;
      corrections[idx][0] = (src[1] - w1 - ot_msg[idx][1]) & mod_mask;
      corrections[idx][1] = (src[2] - w0 - prp_ot_msg[idx][1]) & mod_mask;

      src[1] = w0;
      src[2] = w1;
    }
    {
      int64_t idx = (lvl_p + 2) * (total_numel / 2) + perm_idx;
      auto w0 = (src[0] - ot_msg[idx][0]) & mod_mask;
      auto w1 = (src[1] - prp_ot_msg[idx][0]) & mod_mask;
      corrections[idx][0] = (src[0] - w1 - ot_msg[idx][1]) & mod_mask;
      corrections[idx][1] = (src[1] - w0 - prp_ot_msg[idx][1]) & mod_mask;

      src[0] = w0;
      src[1] = w1;
    }
    return;
  }

  // partea superioara
  for (int64_t i = 0; i < values - 1; i += 2) {
    int64_t idx = lvl_p * (total_numel / 2) + perm_idx + i / 2;
    auto w0 = (src[i + 0] - ot_msg[idx][0]) & mod_mask;
    auto w1 = (src[i + 1] - prp_ot_msg[idx][0]) & mod_mask;
    corrections[idx][0] = (src[i + 0] - w1 - ot_msg[idx][1]) & mod_mask;
    corrections[idx][1] = (src[i + 1] - w0 - prp_ot_msg[idx][1]) & mod_mask;

    src[i + 0] = w0;
    src[i + 1] = w1;

    bottom1.push_back(src[i]);
    top1.push_back(src[i ^ 1]);
  }

  if (values & 1) {
    top1.push_back(src[values - 1]);
  }

  prepare_correction_additive(total_numel, n - 1, lvl_p + 1,
                              perm_idx + values / 4, absl::MakeSpan(top1),
                              corrections, ot_msg, prp_ot_msg);

  prepare_correction_additive(total_numel, n - 1, lvl_p + 1, perm_idx,
                              absl::MakeSpan(bottom1), corrections, ot_msg,
                              prp_ot_msg);

  for (int64_t i = 0; i < values - 1; i += 2) {
    int64_t idx = (lvl_p + levels - 1) * (total_numel / 2) + perm_idx + i / 2;
    auto w0 = (bottom1[i / 2] - ot_msg[idx][0]) & mod_mask;
    auto w1 = (top1[i / 2] - prp_ot_msg[idx][0]) & mod_mask;
    corrections[idx][0] = (bottom1[i / 2] - w1 - ot_msg[idx][1]) & mod_mask;
    corrections[idx][1] = (top1[i / 2] - w0 - prp_ot_msg[idx][1]) & mod_mask;

    src[i + 0] = w0;
    src[i + 1] = w1;
  }

  if (values & 1) {
    src[values - 1] = top1[(values + 1) / 2 - 1];
  }
}

void BenesNetwork::prepare_correction(
    int64_t total_numel, int64_t n, int64_t lvl_p, int64_t perm_idx,
    absl::Span<scalar_t> src, absl::Span<std::array<scalar_t, 2>> corrections,
    absl::Span<const std::array<scalar_t, 2>> ot_msg,
    absl::Span<const std::array<scalar_t, 2>> prp_ot_msg) const {
  // ot message M0 = m0 ^ w0 || m1 ^ w1
  //  for each switch: top wire m0 w0 - bottom wires m1, w1
  //  M1 = m0 ^ w1 || m1 ^ w0
  const int levels = 2 * n - 1;
  const int values = src.size();
  std::vector<scalar_t> bottom1;
  std::vector<scalar_t> top1;

  if (values == 2) {
    if (n == 1) {
      int64_t idx = lvl_p * (total_numel / 2) + perm_idx;
      // rot message = (r0, r1)
      // expand
      //   M0 = r0 | H(r0)
      //   M1 = r1 | H(r1)
      // parse
      //   M0 = m0 ^ w0 | m1 ^ w1
      //   M1 = m0 ^ w1 | m1 ^ w0
      auto w0 = src[0] ^ ot_msg[idx][0];
      auto w1 = src[1] ^ prp_ot_msg[idx][0];
      // share on P2 (bit = 0)
      //   x0 ^ m0 ^ r0
      //   x1 ^ m1 ^ H(r0)
      // share on P2 (bit = 1)
      //   x0 ^ m0 ^ r1
      //   x1 ^ m1 ^ H(r1)
      // find corrections cr0, cr1 such that
      //     cr0 ^ x0 ^ m0 ^ r1 ^ w1 = x0
      //     cr1 ^ x1 ^ m1 ^ H(r1)) ^ w0 = x1
      //  => cr0 = m0 ^ r1 ^ w1
      //  => cr1 = m1 ^ H(r1) ^ w0
      corrections[idx][0] = src[0] ^ ot_msg[idx][1] ^ w1;
      corrections[idx][1] = src[1] ^ prp_ot_msg[idx][1] ^ w0;
      // share on P1 is always w0, w1
      src[0] = w0;
      src[1] = w1;
    } else {
      int64_t idx = (lvl_p + 1) * (total_numel / 2) + perm_idx;
      auto w0 = ot_msg[idx][0] ^ src[0];
      auto w1 = prp_ot_msg[idx][0] ^ src[1];

      corrections[idx][0] = w1 ^ src[0] ^ ot_msg[idx][1];
      corrections[idx][1] = w0 ^ src[1] ^ prp_ot_msg[idx][1];

      src[0] = w0;
      src[1] = w1;
    }
    return;
  }

  if (values == 3) {
    {
      int64_t idx = lvl_p * (total_numel / 2) + perm_idx;
      auto w0 = ot_msg[idx][0] ^ src[0];
      auto w1 = prp_ot_msg[idx][0] ^ src[1];

      corrections[idx][0] = w1 ^ src[0] ^ ot_msg[idx][1];
      corrections[idx][1] = w0 ^ src[1] ^ prp_ot_msg[idx][1];

      src[0] = w0;
      src[1] = w1;
    }
    {
      int64_t idx = (lvl_p + 1) * (total_numel / 2) + perm_idx;
      auto w0 = ot_msg[idx][0] ^ src[1];
      auto w1 = prp_ot_msg[idx][0] ^ src[2];

      corrections[idx][0] = w1 ^ src[1] ^ ot_msg[idx][1];
      corrections[idx][1] = w0 ^ src[2] ^ prp_ot_msg[idx][1];

      src[1] = w0;
      src[2] = w1;
    }
    {
      int64_t idx = (lvl_p + 2) * (total_numel / 2) + perm_idx;
      auto w0 = ot_msg[idx][0] ^ src[0];
      auto w1 = prp_ot_msg[idx][0] ^ src[1];

      corrections[idx][0] = w1 ^ src[0] ^ ot_msg[idx][1];
      corrections[idx][1] = w0 ^ src[1] ^ prp_ot_msg[idx][1];

      src[0] = w0;
      src[1] = w1;
    }
    return;
  }

  // partea superioara
  for (int64_t i = 0; i < values - 1; i += 2) {
    int64_t idx = lvl_p * (total_numel / 2) + perm_idx + i / 2;
    auto w0 = ot_msg[idx][0] ^ src[i + 0];
    auto w1 = prp_ot_msg[idx][0] ^ src[i + 1];

    corrections[idx][0] = w1 ^ src[i + 0] ^ ot_msg[idx][1];
    corrections[idx][1] = w0 ^ src[i + 1] ^ prp_ot_msg[idx][1];

    src[i + 0] = w0;
    src[i + 1] = w1;

    bottom1.push_back(src[i]);
    top1.push_back(src[i ^ 1]);
  }

  if (values & 1) {
    top1.push_back(src[values - 1]);
  }

  prepare_correction(total_numel, n - 1, lvl_p + 1, perm_idx + values / 4,
                     absl::MakeSpan(top1), corrections, ot_msg, prp_ot_msg);

  prepare_correction(total_numel, n - 1, lvl_p + 1, perm_idx,
                     absl::MakeSpan(bottom1), corrections, ot_msg, prp_ot_msg);

  for (int64_t i = 0; i < values - 1; i += 2) {
    int64_t idx = (lvl_p + levels - 1) * (total_numel / 2) + perm_idx + i / 2;
    auto w0 = ot_msg[idx][0] ^ bottom1[i / 2];
    auto w1 = prp_ot_msg[idx][0] ^ top1[i / 2];

    corrections[idx][0] = w1 ^ bottom1[i / 2] ^ ot_msg[idx][1];
    corrections[idx][1] = w0 ^ top1[i / 2] ^ prp_ot_msg[idx][1];

    src[i + 0] = w0;
    src[i + 1] = w1;
  }

  if (values & 1) {
    src[values - 1] = top1[(values + 1) / 2 - 1];
  }
}

void BenesNetwork::eval_with_wire_masks(
    int64_t n, int64_t lvl_p, int64_t perm_idx, absl::Span<scalar_t> src,
    absl::Span<const scalar_t> wire0, absl::Span<const scalar_t> wire1) const {
  const int values = src.size();
  const int width_ = benes_network[0].size();

  if (values == 2) {
    if (n == 1) {
      src[0] ^= wire0[lvl_p * width_ + perm_idx];
      src[1] ^= wire1[lvl_p * width_ + perm_idx];
      if (benes_network[lvl_p][perm_idx] == 1) {
        std::swap(src[0], src[1]);
      }
    } else {
      src[0] ^= wire0[(lvl_p + 1) * width_ + perm_idx];
      src[1] ^= wire1[(lvl_p + 1) * width_ + perm_idx];
      if (benes_network[lvl_p + 1][perm_idx] == 1) {
        std::swap(src[0], src[1]);
      }
    }
    return;
  }

  if (values == 3) {
    src[0] ^= wire0[lvl_p * width_ + perm_idx];
    src[1] ^= wire1[lvl_p * width_ + perm_idx];
    if (benes_network[lvl_p][perm_idx] == 1) {
      std::swap(src[0], src[1]);
    }

    src[1] ^= wire0[(lvl_p + 1) * width_ + perm_idx];
    src[2] ^= wire1[(lvl_p + 1) * width_ + perm_idx];
    if (benes_network[lvl_p + 1][perm_idx] == 1) {
      std::swap(src[1], src[2]);
    }

    src[0] ^= wire0[(lvl_p + 2) * width_ + perm_idx];
    src[1] ^= wire1[(lvl_p + 2) * width_ + perm_idx];
    if (benes_network[lvl_p + 2][perm_idx] == 1) {
      std::swap(src[0], src[1]);
    }

    return;
  }

  std::vector<scalar_t> bottom1;
  std::vector<scalar_t> top1;

  for (int i = 0; i < values - 1; i += 2) {
    int s = benes_network[lvl_p][perm_idx + i / 2];
    src[i + 0] ^= wire0[lvl_p * width_ + perm_idx + i / 2];
    src[i + 1] ^= wire1[lvl_p * width_ + perm_idx + i / 2];

    for (int j = 0; j < 2; ++j) {
      int x = benes_right_cycle_shift((i | j) ^ s, n);
      if (x < values / 2) {
        bottom1.push_back(src[i | j]);
      } else {
        top1.push_back(src[i | j]);
      }
    }
  }
  if (values & 1) {
    top1.push_back(src[values - 1]);
  }

  eval_with_wire_masks(n - 1, lvl_p + 1, perm_idx, absl::MakeSpan(bottom1),
                       wire0, wire1);
  eval_with_wire_masks(n - 1, lvl_p + 1, perm_idx + values / 4,
                       absl::MakeSpan(top1), wire0, wire1);

  const int levels = 2 * n - 1;
  for (int i = 0; i < values - 1; i += 2) {
    int s = benes_network[lvl_p + levels - 1][perm_idx + i / 2];

    for (int j = 0; j < 2; ++j) {
      int x = benes_right_cycle_shift((i | j) ^ s, n);
      if (x < values / 2)
        src[i | j] = bottom1[x];
      else {
        src[i | j] = top1[i / 2];
      }
    }

    auto w0 = wire0[(lvl_p + levels - 1) * width_ + perm_idx + i / 2];
    auto w1 = wire1[(lvl_p + levels - 1) * width_ + perm_idx + i / 2];
    src[i] ^= (s ? w1 : w0);
    src[i ^ 1] ^= (s ? w0 : w1);
  }

  if (values & 1) {
    src[values - 1] = top1[(values + 1) / 2 - 1];
  }
}
