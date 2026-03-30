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

#include "psi/eqt.h"

#include "yacl/crypto/rand/rand.h"
#include "yacl/crypto/tools/prg.h"

#include "libspu/core/type.h"
#include "libspu/mpc/cheetah/nonlinear/equal_prot.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/ot/ot_util.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace psi::mpc {

using namespace spu;
using namespace spu::mpc;
using namespace spu::mpc::cheetah;

static void SetLeafOTMsg(absl::Span<uint8_t> ot_messages, uint8_t digit,
                         uint8_t rnd_eq_bit) {
  size_t N = ot_messages.size();
  SPU_ENFORCE(digit <= N, fmt::format("N={} got digit={}", N, digit));
  std::fill_n(ot_messages.data(), N, rnd_eq_bit);
  for (size_t i = 0; i < N; i++) {
    ot_messages[i] = rnd_eq_bit ^ static_cast<uint8_t>(digit == i);
  }
}

EqTProtocol::EqTProtocol(const std::shared_ptr<BasicOTProtocols>& base,
                         size_t compare_radix)
    : compare_radix_(compare_radix), basic_ot_prot_(base) {
  SPU_ENFORCE(base != nullptr);
  SPU_ENFORCE(compare_radix_ >= 1 && compare_radix_ <= 8);
  is_sender_ = base->Rank() == 0;
}

EqTProtocol::~EqTProtocol() { basic_ot_prot_->Flush(); }

NdArrayRef EqTProtocol::DoCompute(const NdArrayRef& inp, size_t bit_width) {
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
  spu::mpc::cheetah::EqualProtocol base_eq_prot(basic_ot_prot_, width);

  sent = basic_ot_prot_->GetCommunicator()->lctx()->GetStats()->sent_bytes;
  recv = basic_ot_prot_->GetCommunicator()->lctx()->GetStats()->recv_bytes;
  auto out = base_eq_prot.Compute(summed_eq_bits, width)
                 .as(boolean_t)
                 .reshape(inp.shape());
  sent =
      basic_ot_prot_->GetCommunicator()->lctx()->GetStats()->sent_bytes - sent;
  recv =
      basic_ot_prot_->GetCommunicator()->lctx()->GetStats()->recv_bytes - recv;
  return out;
}

NdArrayRef EqTProtocol::Compute(const NdArrayRef& inp, size_t bit_width) {
  return DoCompute(inp, bit_width);
}

}  // namespace psi::mpc
