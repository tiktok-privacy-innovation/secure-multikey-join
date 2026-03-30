// Copyright 2023 Ant Group Co., Ltd.
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

#include "libspu/mpc/cheetah/permute.h"

#include <cstdio>
#include <limits>

#include "yacl/crypto/rand/rand.h"

#include "libspu/mpc/cheetah/nonlinear/osn_prot.h"
#include "libspu/mpc/cheetah/tiled_dispatch.h"
#include "libspu/mpc/cheetah/type.h"
#include "libspu/mpc/utils/permute.h"
#include "libspu/mpc/utils/ring_ops.h"

namespace spu::mpc::cheetah {

NdArrayRef RandPermM::proc(KernelEvalContext* ctx, const Shape& shape) const {
  SPU_ENFORCE_LE(shape.numel(),
                 static_cast<int64_t>(std::numeric_limits<uint32_t>::max()));

  auto type = makeType<PShrTy>();
  NdArrayRef out(type, shape);
  int32_t* perm = out.data<int32_t>();
  std::iota(perm, perm + shape.numel(), 0);

  uint64_t ctr = 0;
  yacl::crypto::ReplayShuffle(perm, perm + shape.numel(),
                              yacl::crypto::RandSeed(/*fast*/ true), &ctr);

  return out.as(type);
}

namespace {

NdArrayRef ApplyPerm(KernelEvalContext* ctx, const NdArrayRef& in,
                     const NdArrayRef& perm, bool is_inv) {
  SPU_ENFORCE_EQ(in.shape(), perm.shape());
  const auto numel = in.numel();

  // We need a compact mem for the permutation
  absl::Span<const int32_t> perm_span;
  std::vector<int32_t> perm_tmp;

  if (perm.isCompact()) {
    perm_span = absl::MakeConstSpan(perm.data<int32_t>(), numel);
  } else {
    perm_tmp.resize(numel);
    NdArrayView<int32_t> xperm(perm);
    for (int64_t i = 0; i < numel; ++i) {
      perm_tmp[i] = xperm[i];
    }
    perm_span = absl::MakeConstSpan(perm_tmp);
  }

  OSNProtocol::Meta osn_meta;
  osn_meta.is_inverse_perm = is_inv;
  osn_meta.is_arithmetic = !in.eltype().isa<BShare>();
  osn_meta.payload_type = in.eltype().as<Ring2k>()->field();
  osn_meta.payload_width = SizeOf(osn_meta.payload_type) * 8;
  if (osn_meta.is_arithmetic) {
    osn_meta.payload_width = in.eltype().as<AShare>()->nbits();
  } else {
    osn_meta.payload_width = in.eltype().as<BShare>()->nbits();
  }
  osn_meta.numel = numel;
  const int rank = ctx->getState<Communicator>()->getRank();
  if (rank == 0 && osn_meta.is_arithmetic) {
    printf("PermA %lld %d\n", in.numel(), osn_meta.payload_width);
  }
  if (rank == 0 && !osn_meta.is_arithmetic) {
    printf("PermB %lld %d\n", in.numel(), osn_meta.payload_width);
  }

  OSNProtocol osn(osn_meta, perm_span);

  auto out = DispatchUnaryFunc(
      ctx, in,
      [&](const NdArrayRef& input,
          const std::shared_ptr<BasicOTProtocols>& ot) {
        // (P_b)^-1 * (P_a)^-1 * P_a * P_b * v
        if (is_inv) {
          if (rank == 1) {
            auto tmp = osn.Send(in, ot);
            return osn.Recv(tmp, ot);
          } else {
            auto tmp = osn.Recv(in, ot);
            return osn.Send(tmp, ot);
          }
        }

        if (rank == 0) {
          auto tmp = osn.Send(in, ot);
          return osn.Recv(tmp, ot);
        } else {
          auto tmp = osn.Recv(in, ot);
          return osn.Send(tmp, ot);
        }
      },
      /*parallel*/ false);

  return out;
}

}  // namespace

NdArrayRef PermAM::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                        const NdArrayRef& perm) const {
  return ApplyPerm(ctx, in, perm, false);
}

NdArrayRef InvPermAM::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                           const NdArrayRef& perm) const {
  return ApplyPerm(ctx, in, perm, true);
}

NdArrayRef PermBM::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                        const NdArrayRef& perm) const {
  return ApplyPerm(ctx, in, perm, false);
}

NdArrayRef InvPermBM::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                           const NdArrayRef& perm) const {
  return ApplyPerm(ctx, in, perm, true);
}

NdArrayRef PermAP::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                        const NdArrayRef& perm) const {
  if (in.numel() > 0) {
    return applyPerm(in, perm);
  }
  return NdArrayRef(in.eltype(), in.shape());
}

NdArrayRef InvPermAP::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                           const NdArrayRef& perm) const {
  if (in.numel() > 0) {
    return applyInvPerm(in, perm);
  }
  return NdArrayRef(in.eltype(), in.shape());
}

NdArrayRef PermBP::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                        const NdArrayRef& perm) const {
  if (in.numel() > 0) {
    return applyPerm(in, perm);
  }
  printf("PermBP ...\n");
  return NdArrayRef(in.eltype(), in.shape());
}

NdArrayRef InvPermBP::proc(KernelEvalContext* ctx, const NdArrayRef& in,
                           const NdArrayRef& perm) const {
  if (in.numel() > 0) {
    return applyInvPerm(in, perm);
  }
  printf("InvPermBP ...\n");
  return NdArrayRef(in.eltype(), in.shape());
}

}  // namespace spu::mpc::cheetah
