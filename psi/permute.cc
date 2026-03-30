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

#include "psi/permute.h"

#include <algorithm>
#include <future>

#include "permute.h"
#include "yacl/crypto/rand/rand.h"
#include "yacl/crypto/tools/prg.h"
#include "yacl/crypto/tools/rp.h"

#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/ot/ot_util.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"

#include "psi/benes_network.h"

using namespace spu::mpc;
using namespace spu::mpc::cheetah;
namespace psi::mpc {

namespace {

template <typename T, typename C>
void PackThenSend(const std::shared_ptr<C> &channel, std::size_t bit_width,
                  absl::Span<const T> msg) {
  std::int64_t num_msg = msg.size();
  if (num_msg == 0) {
    return;
  }

  if (bit_width == 8 * sizeof(T)) {
    channel->template sendAsync<T>(channel->nextRank(), msg, "send:direct");
    return;
  }

  yacl::Buffer bytes_buffer((num_msg * bit_width + 7) / 8);

  auto bytes =
      absl::MakeSpan(bytes_buffer.data<std::uint8_t>(), bytes_buffer.size());

  (void)CompressArrayToBytes<T>(msg, bit_width, bytes);

  channel->template sendAsync<uint8_t>(channel->nextRank(), bytes,
                                       "send:packed");
}

template <typename T, typename C>
void RecvThenUnpack(const std::shared_ptr<C> &channel, std::size_t bit_width,
                    absl::Span<T> msg) {
  std::int64_t num_msg = msg.size();
  if (num_msg == 0) {
    return;
  }

  if (bit_width == 8 * sizeof(T)) {
    auto recv = channel->template recv<T>(channel->nextRank(), "recv:direct");
    SPU_ENFORCE_EQ((int64_t)recv.size(), num_msg);
    std::copy_n(recv.data(), recv.size(), msg.data());
    return;
  }

  auto bytes =
      channel->template recv<uint8_t>(channel->nextRank(), "recv:packed");
  (void)DecompressArrayFromBytes<T>(bytes, bit_width, msg);
}

template <typename T>
void sampleRandMask(absl::Span<T> out, size_t bw) {
  yacl::crypto::Prg<T> prg;
  prg.Fill(out);
  auto mask = makeBitsMask<T>(bw);
  std::transform(out.begin(), out.end(), out.begin(),
                 [&](T x) { return x & mask; });
}

}  // namespace

PermuteBase::PermuteBase(const Meta &meta) : meta_(meta) {
  SPU_ENFORCE(meta.payload_width > 0);
  bn_ = std::make_unique<BenesNetwork>(meta.numel);
  if (meta.is_arithmetic) {
    bn_->set_arithmetic_payload(meta.payload_width);
  } else {
    bn_->set_boolean_payload(meta.payload_width);
  }
}

#define IMPL_SEND(T)                                                           \
  void PermuteSender::Send(absl::Span<const T> input, absl::Span<T> out,       \
                           const ROT &rot) const {                             \
    SPU_ENFORCE_EQ(input.size(), out.size());                                  \
    SPU_ENFORCE_EQ((int64_t)input.size(), meta_.numel);                        \
    const int64_t gates = bn_->total_gates();                                  \
    yacl::Buffer _ot_msg0(gates * sizeof(T));                                  \
    yacl::Buffer _ot_msg1(gates * sizeof(T));                                  \
    absl::Span<T> ot_msg0 = absl::MakeSpan(_ot_msg0.data<T>(), gates);         \
    absl::Span<T> ot_msg1 = absl::MakeSpan(_ot_msg1.data<T>(), gates);         \
    rot->GetSenderCOT()->SendRMCC(ot_msg0, ot_msg1, meta_.payload_width);      \
    yacl::Buffer _input_mask(input.size() * sizeof(T));                        \
    yacl::Buffer _masked_input(input.size() * sizeof(T));                      \
    auto input_mask = absl::MakeSpan(_input_mask.data<T>(), input.size());     \
    auto masked_input = absl::MakeSpan(_masked_input.data<T>(), input.size()); \
    sampleRandMask(input_mask, meta_.payload_width);                           \
    std::copy_n(input_mask.cbegin(), input_mask.size(), out.data());           \
    auto bit_mask = makeBitsMask<T>(meta_.payload_width);                      \
    std::transform(input.cbegin(), input.cend(), input_mask.cbegin(),          \
                   masked_input.data(), [&](T x, T y) -> T {                   \
                     return meta_.is_arithmetic ? (x - y) & bit_mask : x ^ y;  \
                   });                                                         \
    auto comm = rot->GetCommunicator();                                        \
    yacl::Buffer _correction_word(gates * sizeof(T));                          \
    auto correction_word = absl::MakeSpan(_correction_word.data<T>(), gates);  \
    bn_->prepare_correction<T>(out, correction_word, ot_msg0, ot_msg1);        \
    PackThenSend<T>(rot->GetCommunicator(), meta_.payload_width,               \
                    correction_word);                                          \
    PackThenSend<T>(rot->GetCommunicator(), meta_.payload_width,               \
                    masked_input);                                             \
  }

IMPL_SEND(uint8_t)
IMPL_SEND(uint32_t)
IMPL_SEND(uint64_t)
IMPL_SEND(uint128_t)

#undef IMPL_SEND

PermuteSender::PermuteSender(const Meta &meta) : PermuteBase(meta) {}

psi::mpc::PermuteReceiver::PermuteReceiver(const Meta &meta,
                                           absl::Span<const int32_t> perm)
    : PermuteBase(meta) {
  YACL_ENFORCE_EQ(meta.numel, (int64_t)perm.size());
  bn_ = std::make_unique<BenesNetwork>(perm);
  if (meta.is_arithmetic) {
    bn_->set_arithmetic_payload(meta.payload_width);
  } else {
    bn_->set_boolean_payload(meta.payload_width);
  }
}

#define IMPL_RECV(T)                                                           \
  void PermuteReceiver::Recv(absl::Span<const T> input, absl::Span<T> out,     \
                             const ROT &rot) const {                           \
    SPU_ENFORCE_EQ(input.size(), out.size());                                  \
    SPU_ENFORCE_EQ((int64_t)input.size(), meta_.numel);                        \
    const int64_t gates = bn_->total_gates();                                  \
    yacl::Buffer _control(gates);                                              \
    yacl::Buffer _recv_ot_msg(gates * sizeof(T));                              \
    yacl::Buffer _corr_word(gates * sizeof(T));                                \
    auto control = absl::MakeSpan(_control.data<uint8_t>(), gates);            \
    auto recv_ot_msg = absl::MakeSpan(_recv_ot_msg.data<T>(), gates);          \
    auto corr_word = absl::MakeSpan(_corr_word.data<T>(), gates);              \
    bn_->get_control_bits(control);                                            \
    rot->GetReceiverCOT()->RecvRMCC(control, recv_ot_msg,                      \
                                    meta_.payload_width);                      \
    RecvThenUnpack<T>(rot->GetCommunicator(), meta_.payload_width, corr_word); \
    RecvThenUnpack<T>(rot->GetCommunicator(), meta_.payload_width, out);       \
    auto bit_mask = makeBitsMask<T>(meta_.payload_width);                      \
    if (meta_.is_arithmetic) {                                                 \
      for (size_t i = 0; i < input.size(); ++i) {                              \
        out[i] = (out[i] + input[i]) & bit_mask;                               \
      }                                                                        \
      for (int64_t i = 0; i < gates; ++i) {                                    \
        if (control[i]) {                                                      \
          recv_ot_msg[i] = (recv_ot_msg[i] + corr_word[i]) & bit_mask;         \
        }                                                                      \
      }                                                                        \
    } else {                                                                   \
      for (size_t i = 0; i < input.size(); ++i) {                              \
        out[i] = (out[i] ^ input[i]) & bit_mask;                               \
      }                                                                        \
      for (int64_t i = 0; i < gates; ++i) {                                    \
        if (control[i]) {                                                      \
          recv_ot_msg[i] = (recv_ot_msg[i] ^ corr_word[i]) & bit_mask;         \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    bn_->eval_with_masks<T>(out, recv_ot_msg);                                 \
  }

IMPL_RECV(uint8_t)
IMPL_RECV(uint32_t)
IMPL_RECV(uint64_t)
IMPL_RECV(uint128_t)
#undef IMPL_RECV

}  // namespace psi::mpc
