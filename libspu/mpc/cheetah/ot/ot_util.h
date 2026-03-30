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

#pragma once

#include "absl/types/span.h"
#include "yacl/base/int128.h"

#include "libspu/core/ndarray_ref.h"
#include "libspu/core/prelude.h"
#include "libspu/mpc/common/communicator.h"

namespace spu::mpc::cheetah {

template <typename T>
T makeBitsMask(size_t nbits) {
  size_t max = sizeof(T) * 8;
  if (nbits == 0) {
    nbits = max;
  }
  SPU_ENFORCE(nbits <= max);
  T mask = static_cast<T>(-1);
  if (nbits < max) {
    mask = (static_cast<T>(1) << nbits) - 1;
  }
  return mask;
}

template <typename T>
inline T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
size_t ZipArray(absl::Span<const T> inp, size_t bit_width, absl::Span<T> oup) {
  size_t width = sizeof(T) * 8;
  SPU_ENFORCE(bit_width > 0 && width >= bit_width);
  size_t numel = inp.size();
  size_t packed_sze = CeilDiv(numel * bit_width, width);

  SPU_ENFORCE(oup.size() >= packed_sze);

  const T mask = makeBitsMask<T>(bit_width);
  for (size_t i = 0; i < packed_sze; ++i) {
    oup[i] = 0;
  }
  // shift will in [0, 2 * width]
  for (size_t i = 0, has_done = 0; i < numel; i += 1, has_done += bit_width) {
    T real_data = inp[i] & mask;
    size_t packed_index0 = i * bit_width / width;
    size_t shft0 = has_done & (width - 1);
    oup[packed_index0] |= (real_data << shft0);
    if (shft0 + bit_width > width) {
      size_t shft1 = width - shft0;
      size_t packed_index1 = packed_index0 + 1;
      oup[packed_index1] |= (real_data >> shft1);
    }
  }
  return packed_sze;
}

template <typename T>
size_t UnzipArray(absl::Span<const T> inp, size_t bit_width,
                  absl::Span<T> oup) {
  size_t width = sizeof(T) * 8;
  SPU_ENFORCE(bit_width > 0 && bit_width <= width);

  size_t packed_sze = inp.size();
  size_t n = oup.size();
  size_t raw_sze = packed_sze * width / bit_width;
  SPU_ENFORCE(n > 0 && n <= raw_sze);

  const T mask = makeBitsMask<T>(bit_width);
  for (size_t i = 0, has_done = 0; i < n; i += 1, has_done += bit_width) {
    size_t packed_index0 = i * bit_width / width;
    size_t shft0 = has_done & (width - 1);
    oup[i] = (inp[packed_index0] >> shft0);
    if (shft0 + bit_width > width) {
      size_t shft1 = width - shft0;
      size_t packed_index1 = packed_index0 + 1;
      oup[i] |= (inp[packed_index1] << shft1);
    }
    oup[i] &= mask;
  }
  return n;
}

template <typename T>
size_t ZipArrayBit(absl::Span<const T> inp, size_t bit_width,
                   absl::Span<T> oup) {
  return ZipArray<T>(inp, bit_width, oup);
}

template <typename T>
size_t UnzipArrayBit(absl::Span<const T> inp, size_t bit_width,
                     absl::Span<T> oup) {
  return UnzipArray<T>(inp, bit_width, oup);
}

template <typename T>
size_t PackU8Array(absl::Span<const uint8_t> u8array, absl::Span<T> packed) {
  constexpr size_t elsze = sizeof(T);
  const size_t nbytes = u8array.size();
  const size_t numel = CeilDiv(nbytes, elsze);

  SPU_ENFORCE(packed.size() >= numel);

  for (size_t i = 0; i < nbytes; i += elsze) {
    size_t this_batch = std::min(nbytes - i, elsze);
    T acc{0};
    for (size_t j = 0; j < this_batch; ++j) {
      acc = (acc << 8) | u8array[i + j];
    }
    packed[i / elsze] = acc;
  }

  return numel;
}

NdArrayRef OpenShare(const NdArrayRef &shr, ReduceOp op, size_t nbits,
                     std::shared_ptr<Communicator> conn);

uint8_t BoolToU8(absl::Span<const uint8_t> bits);

void U8ToBool(absl::Span<uint8_t> bits, uint8_t u8);

// Taken from emp-tool
// https://github.com/emp-toolkit/emp-tool/blob/master/emp-tool/utils/block.h#L113
void SseTranspose(uint8_t *out, uint8_t const *inp, uint64_t nrows,
                  uint64_t ncols);

std::uint8_t Pack8Bits(absl::Span<const std::uint8_t> b8);

void Unpack8Bits(uint8_t packed, absl::Span<std::uint8_t> b8);

std::uint16_t Pack16Bits(absl::Span<const std::uint8_t> b16);

uint128_t PackAtMost128Bits(absl::Span<const std::uint8_t> bits);

namespace {
template <typename T, typename U>
void decompress_helper(absl::Span<T> dst, std::size_t lshift_amount,
                       absl::Span<const U> src) {
  SPU_ENFORCE(sizeof(T) >= sizeof(U));
  SPU_ENFORCE_EQ(dst.size(), src.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    dst[i] = (static_cast<T>(src[i]) << lshift_amount) | dst[i];
  }
}

template <typename T, typename U>
void compress_helper(absl::Span<const T> src, std::size_t rshift_amount,
                     absl::Span<U> dst) {
  SPU_ENFORCE(sizeof(T) >= sizeof(U));
  SPU_ENFORCE_EQ(src.size(), dst.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    dst[i] = static_cast<U>(src[i] >> rshift_amount);
  }
}
}  // namespace
// Input array = [element0 | element1 | ... ]
// Each element of low-end `k` bits are compactly packed into bytes stream.
// The output buffer length should be larger than `numel * ceil(k/8)` bytes.
//
// Return the number of bytes written into the output buffer.
template <typename T>
size_t CompressArrayToBytes(absl::Span<const T> inp, size_t bit_width,
                            absl::Span<std::uint8_t> oup) {
  const std::size_t width = sizeof(T) * 8;
  const std::size_t total_bytes = (inp.size() * bit_width + 7) / 8;
  const std::size_t numel = inp.size();
  SPU_ENFORCE(bit_width > 0 && width >= bit_width);
  SPU_ENFORCE_GE(oup.size(), total_bytes);
  if (bit_width == width) {
    std::memcpy(oup.data(), inp.data(), total_bytes);
    return total_bytes;
  }

  std::size_t rshift_amount = 0;
  std::size_t bits_to_pack = bit_width;
  if (bits_to_pack >= 64) {
    compress_helper<T, std::uint64_t>(
        inp, rshift_amount,
        absl::MakeSpan(reinterpret_cast<std::uint64_t *>(oup.data()), numel));

    bits_to_pack -= 64;
    rshift_amount += 64;
    oup = oup.subspan(numel * sizeof(std::uint64_t));
  }

  if (bits_to_pack >= 32) {
    compress_helper<T, std::uint32_t>(
        inp, rshift_amount,
        absl::MakeSpan(reinterpret_cast<std::uint32_t *>(oup.data()), numel));

    bits_to_pack -= 32;
    rshift_amount += 32;
    oup = oup.subspan(numel * sizeof(std::uint32_t));
  }

  if (bits_to_pack >= 16) {
    compress_helper<T, std::uint16_t>(
        inp, rshift_amount,
        absl::MakeSpan(reinterpret_cast<std::uint16_t *>(oup.data()), numel));

    bits_to_pack -= 16;
    rshift_amount += 16;
    oup = oup.subspan(numel * sizeof(std::uint16_t));
  }

  if (bits_to_pack >= 8) {
    compress_helper<T, std::uint8_t>(inp, rshift_amount, oup.subspan(0, numel));
    bits_to_pack -= 8;
    rshift_amount += 8;
    oup = oup.subspan(numel * sizeof(std::uint8_t));
  }

  if (bits_to_pack == 0) {
    return total_bytes;
  }

  auto final_mask = makeBitsMask<std::uint8_t>(bits_to_pack);
  auto final_u8 = oup;
  std::fill_n(final_u8.data(), final_u8.size(), 0);

  for (std::size_t i = 0, has_done = 0; i < numel;
       ++i, has_done += bits_to_pack) {
    std::uint8_t real_data = (inp[i] >> rshift_amount) & final_mask;
    std::size_t packed_index0 = i * bits_to_pack / 8;
    std::size_t shft0 = has_done & 7;
    final_u8[packed_index0] |= (real_data << shft0);
    if (shft0 + bits_to_pack > 8) {
      std::size_t shft1 = 8 - shft0;
      std::size_t packed_index1 = packed_index0 + 1;
      final_u8[packed_index1] |= (real_data >> shft1);
    }
  }

  return total_bytes;
}

template <typename T>
std::size_t DecompressArrayFromBytes(absl::Span<const std::uint8_t> inp,
                                     std::size_t bit_width, absl::Span<T> oup) {
  const std::size_t width = sizeof(T) * 8;
  const std::size_t total_bytes = (oup.size() * bit_width + 7) / 8;
  const std::size_t numel = oup.size();

  SPU_ENFORCE(bit_width > 0 && width >= bit_width);
  SPU_ENFORCE_GE(inp.size(), total_bytes);

  if (bit_width == width) {
    std::memcpy(oup.data(), inp.data(), total_bytes);
    return total_bytes;
  }

  std::size_t lshift_amount = 0;
  std::size_t bits_to_unpack = bit_width;
  std::fill_n(oup.data(), numel, 0);
  if (bits_to_unpack >= 64) {
    decompress_helper<T, std::uint64_t>(
        oup, lshift_amount,
        absl::MakeSpan(reinterpret_cast<const std::uint64_t *>(inp.data()),
                       numel));

    bits_to_unpack -= 64;
    lshift_amount += 64;
    inp = inp.subspan(numel * sizeof(std::uint64_t));
  }

  if (bits_to_unpack >= 32) {
    decompress_helper<T, std::uint32_t>(
        oup, lshift_amount,
        absl::MakeSpan(reinterpret_cast<const std::uint32_t *>(inp.data()),
                       numel));

    bits_to_unpack -= 32;
    lshift_amount += 32;
    inp = inp.subspan(numel * sizeof(std::uint32_t));
  }

  if (bits_to_unpack >= 16) {
    decompress_helper<T, std::uint16_t>(
        oup, lshift_amount,
        absl::MakeSpan(reinterpret_cast<const std::uint16_t *>(inp.data()),
                       numel));
    bits_to_unpack -= 16;
    lshift_amount += 16;
    inp = inp.subspan(numel * sizeof(std::uint16_t));
  }

  if (bits_to_unpack >= 8) {
    decompress_helper<T, std::uint8_t>(oup, lshift_amount,
                                       inp.subspan(0, numel));

    bits_to_unpack -= 8;
    lshift_amount += 8;
    inp = inp.subspan(numel * sizeof(std::uint8_t));
  }

  if (bits_to_unpack == 0) {
    return total_bytes;
  }

  // take care the final 7bits
  auto final_mask = makeBitsMask<std::uint8_t>(bits_to_unpack);
  auto final_u8 = inp;

  for (std::size_t i = 0, done = 0; i < numel; ++i, done += bits_to_unpack) {
    std::size_t packed_index0 = i * bits_to_unpack / 8;
    std::size_t shft0 = done & 7;
    std::uint8_t unpack = (final_u8[packed_index0] >> shft0);
    if (shft0 + bits_to_unpack > 8) {
      std::size_t shft1 = 8 - shft0;
      std::size_t packed_index1 = packed_index0 + 1;
      unpack |= (final_u8[packed_index1] << shft1);
    }
    oup[i] = (static_cast<T>(unpack & final_mask) << lshift_amount) | oup[i];
  }

  return total_bytes;
}

}  // namespace spu::mpc::cheetah
