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

#include "libspu/mpc/cheetah/nonlinear/equal_prot.h"

#include "yacl/crypto/rand/rand.h"
#include "yacl/crypto/tools/prg.h"

#include "libspu/core/type.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/ot/ot_util.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

EqualProtocol::EqualProtocol(const std::shared_ptr<BasicOTProtocols>& base,
                             size_t compare_radix)
    : compare_radix_(compare_radix), basic_ot_prot_(base) {
  SPU_ENFORCE(base != nullptr);
  SPU_ENFORCE(compare_radix_ >= 1 && compare_radix_ <= 8);
  is_sender_ = base->Rank() == 0;
}

EqualProtocol::~EqualProtocol() { basic_ot_prot_->Flush(); }

static void SetLeafOTMsg(absl::Span<uint8_t> ot_messages, uint8_t digit,
                         uint8_t rnd_eq_bit) {
  size_t N = ot_messages.size();
  SPU_ENFORCE(digit <= N, fmt::format("N={} got digit={}", N, digit));
  std::fill_n(ot_messages.data(), N, rnd_eq_bit);
  for (size_t i = 0; i < N; i++) {
    ot_messages[i] = rnd_eq_bit ^ static_cast<uint8_t>(digit == i);
  }
}

static void SetPackedLeafOTMsg(absl::Span<uint128_t> ot_messages, uint8_t digit,
                               uint8_t rnd_eq_bit, size_t offset) {
  size_t d = digit;
  size_t N = ot_messages.size();
  SPU_ENFORCE(d <= N, fmt::format("N={} got digit={}", N, digit));
  SPU_ENFORCE(offset < 128, "offset={}", offset);
  for (size_t i = 0; i < N; i++) {
    uint8_t val = rnd_eq_bit ^ static_cast<uint8_t>(d == i);
    ot_messages[i] |= (static_cast<uint128_t>(val) << offset);
  }
}

NdArrayRef EqualProtocol::DoBatchCompute(const NdArrayRef& inp, int64_t numelt,
                                         int64_t bit_width,
                                         int64_t batch_size) {
  auto field = inp.eltype().as<Ring2k>()->field();
  SPU_ENFORCE(batch_size > 0 && batch_size <= 128);
  SPU_ENFORCE(bit_width > 0 && bit_width <= (int)SizeOf(field) * 8);
  if (bit_width % compare_radix_ != 0) {
    bit_width = CeilDiv<int64_t>(bit_width, compare_radix_) * compare_radix_;
  }
  int64_t num_digits = CeilDiv<int64_t>(bit_width, compare_radix_);
  int64_t radix = static_cast<size_t>(1) << compare_radix_;  // one-of-N OT
  int64_t numeq = numelt * batch_size;
  // init to all zero
  std::vector<uint8_t> digits(inp.numel() * num_digits, 0);

  DISPATCH_ALL_FIELDS(field, [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    const auto mask_radix = makeBitsMask<u2k>(compare_radix_);
    NdArrayView<u2k> xinp(inp);

    for (int64_t i = 0; i < inp.numel(); ++i) {
      for (int64_t j = 0; j < num_digits; ++j) {
        uint32_t shft = j * compare_radix_;
        digits[i * num_digits + j] = (xinp[i] >> shft) & mask_radix;
      }
    }
  });

  using ot_msg_t = uint128_t;
  std::vector<uint8_t> leaf_eq(numeq * num_digits, 0);

  if (is_sender_) {
    yacl::crypto::Prg<uint8_t> prg(yacl::crypto::SecureRandSeed());
    prg.Fill(absl::MakeSpan(leaf_eq));
    // convert u8 random to boolean random
    std::transform(leaf_eq.begin(), leaf_eq.end(), leaf_eq.data(),
                   [](uint8_t v) { return static_cast<uint8_t>(v & 1); });

    // n*M instances of 1-of-N OT
    std::vector<ot_msg_t> _leaf_ot_msg(radix * numelt * num_digits, 0);
    auto leaf_ot_msg = absl::MakeSpan(_leaf_ot_msg);

    std::vector<absl::Span<ot_msg_t> > each_leaf_ot_msg(numelt * num_digits);
    for (size_t i = 0; i < each_leaf_ot_msg.size(); ++i) {
      each_leaf_ot_msg[i] = leaf_ot_msg.subspan(i * radix, radix);
    }

    for (int64_t i = 0; i < numelt; ++i) {
      for (int64_t j = 0; j < batch_size; ++j) {
        auto* this_ot_msg = each_leaf_ot_msg.data() + i * num_digits;
        auto* this_digit = digits.data() + (i * batch_size + j) * num_digits;
        auto* this_leaf_eq = leaf_eq.data() + (i * batch_size + j) * num_digits;

        // Step 6, 7 of Alg1 in CF2's paper
        for (int64_t k = 0; k < num_digits; ++k) {
          SetPackedLeafOTMsg(this_ot_msg[k], this_digit[k], this_leaf_eq[k],
                             /*offset*/ j);
        }
      }
    }
    basic_ot_prot_->GetSenderCOT()->SendCMCC(absl::MakeSpan(leaf_ot_msg), radix,
                                             /*bitwidth*/ batch_size);
    basic_ot_prot_->GetSenderCOT()->Flush();
  } else {
    std::vector<ot_msg_t> packed_msg(numelt * num_digits, 0);
    basic_ot_prot_->GetReceiverCOT()->RecvCMCC(
        absl::MakeSpan(digits), radix, absl::MakeSpan(packed_msg), batch_size);

    // extract equality bits from packed messages
    for (int64_t i = 0; i < numelt; ++i) {
      auto* this_pack_msg = packed_msg.data() + i * num_digits;
      for (int64_t j = 0; j < batch_size; ++j) {
        auto* this_leaf_eq = leaf_eq.data() + (i * batch_size + j) * num_digits;
        for (int64_t k = 0; k < num_digits; ++k) {
          this_leaf_eq[k] = static_cast<uint8_t>((this_pack_msg[k] >> j) & 1);
        }
      }
    }
  }

  auto boolean_t = makeType<BShrTy>(field, 1);
  NdArrayRef prev_eq =
      ring_zeros(field, {static_cast<int64_t>(num_digits * numeq)})
          .as(boolean_t);

  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> xprev_eq(prev_eq);
    pforeach(0, xprev_eq.numel(), [&](int64_t i) { xprev_eq[i] = leaf_eq[i]; });
  });

  Shape oshape = inp.shape();
  if (not is_sender_) {
    oshape.insert(oshape.end(), batch_size);
  }

  return TraversalAND(prev_eq, numeq, num_digits).as(boolean_t).reshape(oshape);
}

NdArrayRef EqualProtocol::TraversalANDFullBinaryTree(NdArrayRef eq,
                                                     size_t num_input,
                                                     size_t num_digits) {
  SPU_ENFORCE(num_digits > 0 && absl::has_single_bit(num_digits),
              "require num_digits be a 2-power");
  if (num_digits == 1) {
    return eq;
  }
  SPU_ENFORCE(eq.shape().size() == 1, "need 1D array");
  SPU_ENFORCE_EQ(num_input * num_digits, (size_t)eq.numel());

  for (size_t i = 1; i <= num_digits; i += 1) {
    int64_t current_num_digits = num_digits / (1 << (i - 1));
    if (current_num_digits == 1) {
      break;
    }
    // eq[i-1, j] <- eq[i, 2*j] * eq[i, 2*j+1]
    int64_t n = current_num_digits * num_input;
    auto lhs_eq = eq.slice({0}, {n}, {2});
    auto rhs_eq = eq.slice({1}, {n}, {2});
    eq = basic_ot_prot_->BitwiseAnd(rhs_eq, lhs_eq);
  }

  return eq;
}

NdArrayRef EqualProtocol::TraversalAND(NdArrayRef eq, size_t num_input,
                                       size_t num_digits) {
  if (absl::has_single_bit(num_digits)) {
    return TraversalANDFullBinaryTree(eq, num_input, num_digits);
  }

  // Split the current tree into two subtrees
  size_t current_num_digits = absl::bit_floor(num_digits);

  Shape current_shape({static_cast<int64_t>(current_num_digits * num_input)});
  NdArrayRef current_eq(eq.eltype(), current_shape);
  // Copy from the CMP and EQ bits for the current sub-full-tree
  pforeach(0, num_input, [&](int64_t i) {
    std::memcpy(&current_eq.at(i * current_num_digits), &eq.at(i * num_digits),
                current_num_digits * eq.elsize());
  });

  NdArrayRef subtree_eq =
      TraversalANDFullBinaryTree(current_eq, num_input, current_num_digits);

  // NOTE(lwj): +1 due to the AND on the sub-full-tree
  size_t remain_num_digits = num_digits - current_num_digits + 1;
  while (remain_num_digits > 1) {
    current_num_digits = absl::bit_floor(remain_num_digits);
    Shape current_shape({static_cast<int64_t>(current_num_digits * num_input)});
    NdArrayRef current_eq(eq.eltype(), current_shape);

    pforeach(0, num_input, [&](int64_t i) {
      // copy subtree result as the 1st digit
      std::memcpy(&current_eq.at(i * current_num_digits), &subtree_eq.at(i),
                  1 * eq.elsize());
      // copy the remaining digits from the input 'cmp' and 'eq'
      std::memcpy(&current_eq.at(i * current_num_digits + 1),
                  &eq.at((i + 1) * num_digits - remain_num_digits + 1),
                  (current_num_digits - 1) * eq.elsize());
    });

    // NOTE(lwj): current_num_digits is not a 2-power
    subtree_eq = TraversalAND(current_eq, num_input, current_num_digits);
    remain_num_digits = remain_num_digits - current_num_digits + 1;
  }

  return subtree_eq;
}

NdArrayRef EqualProtocol::DoFlattenVersion(const NdArrayRef& inp,
                                           size_t bit_width) {
  bool verbose = false;
  auto field = inp.eltype().as<Ring2k>()->field();
  if (bit_width == 0) {
    bit_width = SizeOf(field) * 8;
  }
  bit_width = std::min(bit_width, SizeOf(field) * 8);

  int64_t remain = bit_width % compare_radix_;
  int64_t num_digits = CeilDiv(bit_width, compare_radix_);
  int64_t radix = static_cast<size_t>(1) << compare_radix_;  // one-of-N OT
  int64_t num_eq = inp.numel();
  // init to all zero
  std::vector<uint8_t> digits(num_eq * num_digits, 0);

  DISPATCH_ALL_FIELDS(field, [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    const auto mask_radix = makeBitsMask<u2k>(compare_radix_);
    const auto mask_remain = makeBitsMask<u2k>(remain);
    NdArrayView<u2k> xinp(inp);

    for (int64_t i = 0; i < num_eq; ++i) {
      for (int64_t j = 0; j < num_digits; ++j) {
        uint32_t shft = j * compare_radix_;
        digits[i * num_digits + j] = (xinp[i] >> shft) & mask_radix;
        // last digits
        if (remain > 0 && (j + 1 == num_digits)) {
          digits[i * num_digits + j] &= mask_remain;
        }
      }
    }
  });

  std::vector<uint8_t> leaf_eq(num_eq * num_digits, 0);
  size_t sent =
      basic_ot_prot_->GetCommunicator()->lctx()->GetStats()->sent_bytes;
  size_t recv =
      basic_ot_prot_->GetCommunicator()->lctx()->GetStats()->recv_bytes;
  if (is_sender_) {
    yacl::crypto::Prg<uint8_t> prg(yacl::crypto::SecureRandSeed());
    prg.Fill(absl::MakeSpan(leaf_eq));
    // convert u8 random to boolean random
    std::transform(leaf_eq.begin(), leaf_eq.end(), leaf_eq.data(),
                   [](uint8_t v) { return v & 1; });

    // n*M instances of 1-of-N OT
    std::vector<uint8_t> leaf_ot_msg(radix * num_eq * num_digits, 0);

    std::vector<absl::Span<uint8_t> > each_leaf_ot_msg(num_eq * num_digits);
    for (size_t i = 0; i < each_leaf_ot_msg.size(); ++i) {
      each_leaf_ot_msg[i] =
          absl::MakeSpan(leaf_ot_msg.data() + i * radix, radix);
    }

    for (int64_t i = 0; i < num_eq; ++i) {
      auto* this_ot_msg = each_leaf_ot_msg.data() + i * num_digits;
      auto* this_digit = digits.data() + i * num_digits;
      auto* this_leaf_eq = leaf_eq.data() + i * num_digits;

      // Step 6, 7 of Alg1 in CF2's paper
      for (int64_t j = 0; j < num_digits; ++j) {
        uint8_t rnd_eq = this_leaf_eq[j] & 1;
        SetLeafOTMsg(this_ot_msg[j], this_digit[j], rnd_eq);
      }
    }

    basic_ot_prot_->GetSenderCOT()->SendCMCC(absl::MakeSpan(leaf_ot_msg), radix,
                                             /*bitwidth*/ 1);
    basic_ot_prot_->GetSenderCOT()->Flush();
  } else {
    basic_ot_prot_->GetReceiverCOT()->RecvCMCC(absl::MakeSpan(digits), radix,
                                               absl::MakeSpan(leaf_eq), 1);
  }
  sent =
      basic_ot_prot_->GetCommunicator()->lctx()->GetStats()->sent_bytes - sent;
  recv =
      basic_ot_prot_->GetCommunicator()->lctx()->GetStats()->recv_bytes - recv;

  if (is_sender_ && verbose) {
    printf("Batch OoNOT %f bit per\n", (sent + recv) * 8. / num_eq);
  }

  auto boolean_t = makeType<BShrTy>(field, 1);
  NdArrayRef prev_eq =
      ring_zeros(field, {static_cast<int64_t>(num_digits * num_eq)})
          .as(boolean_t);

  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> xprev_eq(prev_eq);
    pforeach(0, xprev_eq.numel(), [&](int64_t i) { xprev_eq[i] = leaf_eq[i]; });
  });

  size_t width = absl::bit_width(static_cast<size_t>(num_digits));

  sent = basic_ot_prot_->GetCommunicator()->lctx()->GetStats()->sent_bytes;
  recv = basic_ot_prot_->GetCommunicator()->lctx()->GetStats()->recv_bytes;

  auto eq_bits = basic_ot_prot_->B2ASingleBitWithSize(prev_eq, width);

  sent =
      basic_ot_prot_->GetCommunicator()->lctx()->GetStats()->sent_bytes - sent;
  recv =
      basic_ot_prot_->GetCommunicator()->lctx()->GetStats()->recv_bytes - recv;

  // if (is_sender_ && verbose) {
  //   printf("B2A width %zd digits %lld, exchange %f + %f bit per\n", width,
  //          num_digits, (sent) * 8. / num_eq, (recv) * 8. / num_eq);
  // }

  auto arith_t = makeType<AShrTy>(field);
  NdArrayRef summed_eq_bits =
      ring_zeros(field, {static_cast<int64_t>(num_eq)}).as(arith_t);

  // a0 + b0 = N mod 2^k
  // a0 = N - b0
  DISPATCH_ALL_FIELDS(field, [&]() {
    ring2k_t msk = makeBitsMask<ring2k_t>(width);
    NdArrayView<ring2k_t> out(summed_eq_bits);
    NdArrayView<const ring2k_t> inp(eq_bits);
    for (int64_t i = 0; i < num_eq; ++i) {
      for (int64_t j = 0; j < num_digits; ++j) {
        out[i] = (out[i] + inp[i * num_digits + j]) & msk;
      }

      if (is_sender_) {
        out[i] = (num_digits - out[i]) & msk;
      }
    }
  });

  // 1-of-2^k OT
  // return summed_eq_bits;
  // 1-of-2^k
  // [A] == [0]

  EqualProtocol eq_prot(basic_ot_prot_, width);

  sent = basic_ot_prot_->GetCommunicator()->lctx()->GetStats()->sent_bytes;
  recv = basic_ot_prot_->GetCommunicator()->lctx()->GetStats()->recv_bytes;
  auto out = eq_prot.DoCompute(summed_eq_bits, width)
                 .as(boolean_t)
                 .reshape(inp.shape());
  sent =
      basic_ot_prot_->GetCommunicator()->lctx()->GetStats()->sent_bytes - sent;
  recv =
      basic_ot_prot_->GetCommunicator()->lctx()->GetStats()->recv_bytes - recv;
  if (is_sender_ && verbose) {
    printf("Single OoNOT %f bit per\n", (sent + recv) * 8. / num_eq);
  }
  return out;
}

NdArrayRef EqualProtocol::DoCompute(const NdArrayRef& inp, size_t bit_width) {
  auto field = inp.eltype().as<Ring2k>()->field();
  if (bit_width == 0) {
    bit_width = SizeOf(field) * 8;
  }
  bit_width = std::min(bit_width, SizeOf(field) * 8);

  int64_t remain = bit_width % compare_radix_;
  int64_t num_digits = CeilDiv(bit_width, compare_radix_);
  int64_t radix = static_cast<size_t>(1) << compare_radix_;  // one-of-N OT
  int64_t num_eq = inp.numel();
  // init to all zero
  std::vector<uint8_t> digits(num_eq * num_digits, 0);

  DISPATCH_ALL_FIELDS(field, [&]() {
    using u2k = std::make_unsigned<ring2k_t>::type;
    const auto mask_radix = makeBitsMask<u2k>(compare_radix_);
    const auto mask_remain = makeBitsMask<u2k>(remain);
    NdArrayView<u2k> xinp(inp);

    for (int64_t i = 0; i < num_eq; ++i) {
      for (int64_t j = 0; j < num_digits; ++j) {
        uint32_t shft = j * compare_radix_;
        digits[i * num_digits + j] = (xinp[i] >> shft) & mask_radix;
        // last digits
        if (remain > 0 && (j + 1 == num_digits)) {
          digits[i * num_digits + j] &= mask_remain;
        }
      }
    }
  });

  size_t sent =
      basic_ot_prot_->GetCommunicator()->lctx()->GetStats()->sent_bytes;
  size_t recv =
      basic_ot_prot_->GetCommunicator()->lctx()->GetStats()->recv_bytes;
  std::vector<uint8_t> leaf_eq(num_eq * num_digits, 0);
  if (is_sender_) {
    yacl::crypto::Prg<uint8_t> prg(yacl::crypto::SecureRandSeed());
    prg.Fill(absl::MakeSpan(leaf_eq));
    // convert u8 random to boolean random
    std::transform(leaf_eq.begin(), leaf_eq.end(), leaf_eq.data(),
                   [](uint8_t v) { return v & 1; });

    // n*M instances of 1-of-N OT
    std::vector<uint8_t> leaf_ot_msg(radix * num_eq * num_digits, 0);

    std::vector<absl::Span<uint8_t> > each_leaf_ot_msg(num_eq * num_digits);
    for (size_t i = 0; i < each_leaf_ot_msg.size(); ++i) {
      each_leaf_ot_msg[i] =
          absl::MakeSpan(leaf_ot_msg.data() + i * radix, radix);
    }

    for (int64_t i = 0; i < num_eq; ++i) {
      auto* this_ot_msg = each_leaf_ot_msg.data() + i * num_digits;
      auto* this_digit = digits.data() + i * num_digits;
      auto* this_leaf_eq = leaf_eq.data() + i * num_digits;

      // Step 6, 7 of Alg1 in CF2's paper
      for (int64_t j = 0; j < num_digits; ++j) {
        uint8_t rnd_eq = this_leaf_eq[j] & 1;
        SetLeafOTMsg(this_ot_msg[j], this_digit[j], rnd_eq);
      }
    }

    basic_ot_prot_->GetSenderCOT()->SendCMCC(absl::MakeSpan(leaf_ot_msg), radix,
                                             /*bitwidth*/ 1);
    basic_ot_prot_->GetSenderCOT()->Flush();
  } else {
    basic_ot_prot_->GetReceiverCOT()->RecvCMCC(absl::MakeSpan(digits), radix,
                                               absl::MakeSpan(leaf_eq), 1);
  }
  sent =
      basic_ot_prot_->GetCommunicator()->lctx()->GetStats()->sent_bytes - sent;
  recv =
      basic_ot_prot_->GetCommunicator()->lctx()->GetStats()->recv_bytes - recv;

  auto boolean_t = makeType<BShrTy>(field, 1);
  NdArrayRef prev_eq =
      ring_zeros(field, {static_cast<int64_t>(num_digits * num_eq)})
          .as(boolean_t);

  DISPATCH_ALL_FIELDS(field, [&]() {
    NdArrayView<ring2k_t> xprev_eq(prev_eq);
    pforeach(0, xprev_eq.numel(), [&](int64_t i) { xprev_eq[i] = leaf_eq[i]; });
  });

  sent = basic_ot_prot_->GetCommunicator()->lctx()->GetStats()->sent_bytes;
  recv = basic_ot_prot_->GetCommunicator()->lctx()->GetStats()->recv_bytes;
  auto ret = TraversalAND(prev_eq, num_eq, num_digits)
                 .as(boolean_t)
                 .reshape(inp.shape());
  sent =
      basic_ot_prot_->GetCommunicator()->lctx()->GetStats()->sent_bytes - sent;
  recv =
      basic_ot_prot_->GetCommunicator()->lctx()->GetStats()->recv_bytes - recv;

  return ret;
}

NdArrayRef EqualProtocol::Compute(const NdArrayRef& inp, size_t bit_width) {
  return DoCompute(inp, bit_width);
}

NdArrayRef EqualProtocol::FlattedCompute(const NdArrayRef& inp,
                                         size_t bit_width) {
  return DoFlattenVersion(inp, bit_width);
}

NdArrayRef EqualProtocol::BatchCompute(const NdArrayRef& inp, int64_t numel,
                                       int64_t bitwidth, int64_t batch_size) {
  int64_t bw = SizeOf(inp.eltype().as<Ring2k>()->field()) * 8;
  SPU_ENFORCE(bitwidth >= 0 && bitwidth <= bw, "bit_width={} out of bound",
              bitwidth);
  SPU_ENFORCE(batch_size >= 1 && batch_size <= 128,
              "batch_size={} out of bound", batch_size);
  if (is_sender_) {
    SPU_ENFORCE_EQ(inp.numel(), numel * batch_size);
  } else {
    SPU_ENFORCE_EQ(inp.numel(), numel);
  }

  if (bitwidth == 0) {
    bitwidth = bw;
  }

  return DoBatchCompute(inp, numel, bitwidth, batch_size);
}

}  // namespace spu::mpc::cheetah
