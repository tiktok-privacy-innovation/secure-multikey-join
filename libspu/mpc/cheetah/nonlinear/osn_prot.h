#pragma once

#include <memory>

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/shape.h"
#include "libspu/core/type_util.h"

namespace spu::mpc::cheetah {

class BasicOTProtocols;

class BenesNetwork;

class OSNProtocol {
 public:
  // md5("OSN-FERRET_OT-1-0-0")
  constexpr static uint64_t kPRPKey[2] = {0x0193810b5735cc87,
                                          0x2ae40a13041ccc87};

  struct Meta {
    bool is_inverse_perm;
    bool is_arithmetic;
    int32_t payload_width;
    FieldType payload_type;

    int64_t numel;
  };

  explicit OSNProtocol(const Meta &meta, absl::Span<const int32_t> permutation);

  explicit OSNProtocol(const Meta &meta);

  ~OSNProtocol() = default;

  // P1 (sender) provides an input vector x
  // P2 (receiver) provides a permutation pi
  // gives pi(x) - y to P1, and gives y to P2
  NdArrayRef Send(const NdArrayRef &inp,
                  const std::shared_ptr<BasicOTProtocols> &base) const;
  // P2 should know the length of 'x'
  NdArrayRef Recv(int64_t numel,
                  const std::shared_ptr<BasicOTProtocols> &base) const {
    return Recv(Shape({numel}), base);
  }

  NdArrayRef Recv(Shape shape,
                  const std::shared_ptr<BasicOTProtocols> &base) const;
  // A secret share variant where the input vector is already shared between P1
  // and P2.
  // Specifically, P1 provides x0 and P2 provides pi and x1
  // such that x = x0 + x1.
  // gives pi(x) - y to P1, and gives y to P2.
  NdArrayRef Recv(const NdArrayRef &input_share,
                  const std::shared_ptr<BasicOTProtocols> &base) const;

  // [mask, msg]
  std::array<NdArrayRef, 2> SendPartial(
      Shape shape, const std::shared_ptr<BasicOTProtocols> &base) const;

  NdArrayRef RecvPartial(Shape shape,
                         const std::shared_ptr<BasicOTProtocols> &base);

  void FinishSend(const NdArrayRef &msg, const NdArrayRef &mask,
                  const std::shared_ptr<BasicOTProtocols> &base) const;

  NdArrayRef FinishRecv(const NdArrayRef &inp, const NdArrayRef &recv_ot_msg,
                        const std::shared_ptr<BasicOTProtocols> &base) const;

  NdArrayRef FinishRecv(Shape shape, const NdArrayRef &recv_ot_msg,
                        const std::shared_ptr<BasicOTProtocols> &base) const;

 private:
  Meta meta_;
  std::shared_ptr<BenesNetwork> bn_ = nullptr;
};

}  // namespace spu::mpc::cheetah
