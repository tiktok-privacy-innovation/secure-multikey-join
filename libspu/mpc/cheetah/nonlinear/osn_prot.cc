#include "libspu/mpc/cheetah/nonlinear/osn_prot.h"

#include <algorithm>
#include <future>

#include "yacl/crypto/rand/rand.h"
#include "yacl/crypto/tools/prg.h"
#include "yacl/crypto/tools/rp.h"

#include "libspu/mpc/cheetah/nonlinear/benes_network.h"
#include "libspu/mpc/cheetah/ot/basic_ot_prot.h"
#include "libspu/mpc/cheetah/ot/ot_util.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

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

NdArrayRef SampleRandMask(FieldType ft, Shape shape, size_t bw) {
  auto x = ring_rand(ft, shape);

  DISPATCH_ALL_FIELDS(ft, [&]() {
    NdArrayView<ring2k_t> xwrap(x);
    auto mask = makeBitsMask<ring2k_t>(bw);
    pforeach(0, x.numel(), [&](int64_t i) { xwrap[i] &= mask; });
  });
  return x;
}

}  // namespace

OSNProtocol::OSNProtocol(const Meta &meta,
                         absl::Span<const int32_t> permutation)
    : meta_(meta) {
  SPU_ENFORCE_EQ(meta.numel, (int64_t)permutation.size());
  SPU_ENFORCE(meta.payload_width > 0);
  SPU_ENFORCE_GE(SizeOf(meta.payload_type) * 8, (size_t)meta.payload_width);
  bn_ = std::make_shared<BenesNetwork>(permutation);
  if (meta.is_arithmetic) {
    bn_->set_arithmetic_payload(meta.payload_width);
  } else {
    bn_->set_boolean_payload(meta.payload_width);
  }
}

OSNProtocol::OSNProtocol(const Meta &meta) : meta_(meta) {
  SPU_ENFORCE(meta.payload_width > 0);
  SPU_ENFORCE_GE(SizeOf(meta.payload_type) * 8, (size_t)meta.payload_width);
  bn_ = std::make_shared<BenesNetwork>(meta.numel);
  if (meta.is_arithmetic) {
    bn_->set_arithmetic_payload(meta.payload_width);
  } else {
    bn_->set_boolean_payload(meta.payload_width);
  }
}

NdArrayRef OSNProtocol::Recv(
    Shape shape, const std::shared_ptr<BasicOTProtocols> &base) const {
  SPU_ENFORCE_EQ(meta_.numel, shape.numel());
  const int64_t gates = bn_->total_gates();
  std::vector<uint8_t> bits(gates);
  bn_->get_control_bits(absl::MakeSpan(bits));

  return DISPATCH_ALL_FIELDS(meta_.payload_type, [&]() {
    std::vector<ring2k_t> _recv_ot_msg(gates);
    auto recv_ot_msg = absl::MakeSpan(_recv_ot_msg);

    base->GetReceiverCOT()->RecvRMCC(absl::MakeSpan(bits), recv_ot_msg,
                                     meta_.payload_width);
    std::vector<ring2k_t> corr(gates);
    std::vector<ring2k_t> masked_input(shape.numel());
    auto comm = base->GetCommunicator();

    auto io_task = std::async([&]() {
      RecvThenUnpack<ring2k_t>(comm, meta_.payload_width, absl::MakeSpan(corr));
      RecvThenUnpack<ring2k_t>(comm, meta_.payload_width,
                               absl::MakeSpan(masked_input));
    });

    // wait IO
    io_task.get();

    if (meta_.is_arithmetic) {
      auto msk = makeBitsMask<ring2k_t>(meta_.payload_width);
      pforeach(0, gates, [&](int64_t i) {
        if (bits[i]) {
          recv_ot_msg[i] = (recv_ot_msg[i] + corr[i]) & msk;
        }
      });

      bn_->eval_with_masks<ring2k_t>(absl::MakeSpan(masked_input), recv_ot_msg);
    } else {
      pforeach(0, gates, [&](int64_t i) {
        if (bits[i]) {
          recv_ot_msg[i] ^= corr[i];
        }
      });

      bn_->eval_with_masks<ring2k_t>(absl::MakeSpan(masked_input), recv_ot_msg);
    }

    NdArrayRef out = ring_zeros(meta_.payload_type, shape);
    std::copy_n(masked_input.data(), masked_input.size(), out.data<ring2k_t>());
    auto btype = makeType<BShrTy>(meta_.payload_type);
    auto atype = makeType<AShrTy>(meta_.payload_type);
    return out.as(meta_.is_arithmetic ? atype : btype);
  });
}

NdArrayRef OSNProtocol::Recv(
    const NdArrayRef &inp,
    const std::shared_ptr<BasicOTProtocols> &base) const {
  SPU_ENFORCE(inp.eltype().isa<RingTy>());
  auto ft = inp.eltype().as<RingTy>()->field();
  SPU_ENFORCE_EQ(meta_.numel, inp.numel());
  SPU_ENFORCE_EQ(meta_.payload_type, ft);

  const int64_t gates = bn_->total_gates();
  std::vector<uint8_t> bits(gates);
  bn_->get_control_bits(absl::MakeSpan(bits));

  return DISPATCH_ALL_FIELDS(ft, [&]() {
    std::vector<ring2k_t> _recv_ot_msg(gates);
    auto recv_ot_msg = absl::MakeSpan(_recv_ot_msg);
    base->GetReceiverCOT()->RecvRMCC(absl::MakeSpan(bits), recv_ot_msg,
                                     meta_.payload_width);
    std::vector<ring2k_t> corr(gates);
    std::vector<ring2k_t> masked_input(inp.numel());
    auto comm = base->GetCommunicator();

    auto io_task = std::async([&]() {
      RecvThenUnpack<ring2k_t>(comm, meta_.payload_width, absl::MakeSpan(corr));
      RecvThenUnpack<ring2k_t>(comm, meta_.payload_width,
                               absl::MakeSpan(masked_input));
    });

    // wait IO
    io_task.get();

    NdArrayView<ring2k_t> input_share(inp);
    if (meta_.is_arithmetic) {
      auto msk = makeBitsMask<ring2k_t>(meta_.payload_width);
      pforeach(0, inp.numel(), [&](int64_t i) {
        masked_input[i] = (masked_input[i] + input_share[i]) & msk;
      });

      pforeach(0, gates, [&](int64_t i) {
        if (bits[i]) {
          recv_ot_msg[i] = (recv_ot_msg[i] + corr[i]) & msk;
        }
      });
      bn_->eval_with_masks<ring2k_t>(absl::MakeSpan(masked_input), recv_ot_msg);

    } else {
      auto msk = makeBitsMask<ring2k_t>(meta_.payload_width);
      pforeach(0, inp.numel(), [&](int64_t i) {
        masked_input[i] = (masked_input[i] ^ input_share[i]) & msk;
      });

      pforeach(0, gates, [&](int64_t i) {
        if (bits[i]) {
          // b ^ c
          recv_ot_msg[i] = (recv_ot_msg[i] ^ corr[i]) & msk;
        }
      });

      bn_->eval_with_masks<ring2k_t>(absl::MakeSpan(masked_input), recv_ot_msg);
    }

    NdArrayRef out = ring_zeros(meta_.payload_type, inp.shape());
    std::copy_n(masked_input.data(), masked_input.size(), out.data<ring2k_t>());
    return out.as(inp.eltype());
  });
}

NdArrayRef OSNProtocol::Send(
    const NdArrayRef &inp,
    const std::shared_ptr<BasicOTProtocols> &base) const {
  SPU_ENFORCE(inp.eltype().isa<RingTy>());
  auto ft = inp.eltype().as<RingTy>()->field();
  SPU_ENFORCE_EQ(meta_.numel, inp.numel());
  SPU_ENFORCE_EQ(meta_.payload_type, ft);

  return DISPATCH_ALL_FIELDS(ft, [&]() {
    const int64_t gates = bn_->total_gates();

    std::vector<ring2k_t> _ot_msg0(gates);
    std::vector<ring2k_t> _ot_msg1(gates);
    auto ot_msg0 = absl::MakeSpan(_ot_msg0);
    auto ot_msg1 = absl::MakeSpan(_ot_msg1);

    base->GetSenderCOT()->SendRMCC(ot_msg0, ot_msg1, meta_.payload_width);

    NdArrayRef input_mask =
        SampleRandMask(ft, inp.shape(), meta_.payload_width);
    NdArrayRef updated_mask = input_mask.clone();

    std::vector<ring2k_t> _cw0(gates);
    auto cw = absl::MakeSpan(_cw0);

    bn_->prepare_correction<ring2k_t>(
        absl::MakeSpan(updated_mask.data<ring2k_t>(), inp.numel()), cw, ot_msg0,
        ot_msg1);

    auto comm = base->GetCommunicator();
    auto masked_input = meta_.is_arithmetic ? ring_sub(inp, input_mask)
                                            : ring_xor(inp, input_mask);
    auto minp =
        absl::MakeSpan(masked_input.data<ring2k_t>(), masked_input.numel());

    PackThenSend<ring2k_t>(comm, meta_.payload_width, cw);
    PackThenSend<ring2k_t>(comm, meta_.payload_width, minp);

    return updated_mask.as(inp.eltype());
  });
}

}  // namespace spu::mpc::cheetah
