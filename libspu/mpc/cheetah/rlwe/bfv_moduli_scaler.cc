#include "libspu/mpc/cheetah/rlwe/bfv_moduli_scaler.h"

#include <algorithm>
#include <stdexcept>

#include "bfv_moduli_scaler.h"
#include "seal/util/iterator.h"
#include "seal/util/numth.h"
#include "seal/util/polyarithsmallmod.h"
#include "seal/util/uintarith.h"

inline static u64 U64BitMask(std::size_t bw) {
  // SPU_ENFORCE(bw > 0 && bw <= 64);
  return bw == 64 ? static_cast<u64>(-1) : (static_cast<u64>(1) << bw) - 1;
}

// return x^-1 mod 2^k for odd x
template <typename T>
constexpr T inv_mod_2k(const T &x) {
  if (not(x & 1)) {
    throw std::invalid_argument("need odd input");
  }
  constexpr int nbits = sizeof(T) * 8;
  T inv = 1;
  T p = x;
  for (int i = 1; i < nbits; ++i) {
    inv *= p;
    p *= p;
  }
  return inv;
}

// x mod prime
template <typename T>
u64 barrett_reduce(T x, const seal::Modulus &prime) {
  if constexpr (std::is_same_v<T, uint128_t>) {
    u64 z[2]{static_cast<u64>(x), static_cast<u64>(x >> 64)};
    return seal::util::barrett_reduce_128(z, prime);
  } else {
    return seal::util::barrett_reduce_64(static_cast<u64>(x), prime);
  }
}

static u128 limbs_mod_2k(const u64 *limbs, std::size_t size, u32 mod_bitlen) {
  // SPU_ENFORCE(mod_bitlen <= 128 && mod_bitlen >= 2);

  u32 num_limbs = (mod_bitlen + 63) / 64;
  u32 msb = mod_bitlen - 64 * (num_limbs - 1);
  u64 Q_mod_t_lo = limbs[0];
  u64 Q_mod_t_hi = size > 1 ? limbs[1] : 0;
  if (num_limbs > 1) {
    Q_mod_t_hi &= U64BitMask(msb);
  } else {
    Q_mod_t_hi = 0;
    Q_mod_t_lo &= U64BitMask(msb);
  }
  return (static_cast<u128>(Q_mod_t_hi) << 64) | Q_mod_t_lo;
}

BFVModuliScaler::BFVModuliScaler(const seal::SEALContext &context,
                                 u32 plain_modulus_bitlen)
    : context_(context) {
  // check plain_modulus_bitlen
  initialize_mod_2k(plain_modulus_bitlen);
}

void BFVModuliScaler::initialize_mod_2k(u32 plain_modulus_bitlen) {
  plain_modulus_bitlen_ = plain_modulus_bitlen;

  const auto cntxt_dat = context_.key_context_data();
  u32 logQ = cntxt_dat->total_coeff_modulus_bit_count();
  std::size_t poly_degree = cntxt_dat->parms().poly_modulus_degree();
  const auto &coeff_modulus = cntxt_dat->parms().coeff_modulus();
  std::size_t num_modulus = coeff_modulus.size();
  const int gamma_bits = SEAL_INTERNAL_MOD_BIT_COUNT;
  gamma_ = seal::util::get_primes(poly_degree, gamma_bits, 1)[0];
  for (const auto &modulus : coeff_modulus) {
    if (gamma_.value() == modulus.value()) {
      throw std::runtime_error("gamma");
    }
  }

  mod_plain_mask_ = (static_cast<u128>(1) << plain_modulus_bitlen_) - 1;

  plain_half_ = static_cast<u128>(1) << (plain_modulus_bitlen_ - 1);

  const u64 *Q_limbs = cntxt_dat->total_coeff_modulus();

  std::vector<u64> Q_div_plain(num_modulus, 0);  // Q div 2^k
  if (logQ > plain_modulus_bitlen_) {
    seal::util::right_shift_uint(Q_limbs, plain_modulus_bitlen_, num_modulus,
                                 Q_div_plain.data());
  } else {
    std::fill_n(Q_div_plain.data(), num_modulus, 0);
  }

  Q_mod_plain_ = limbs_mod_2k(Q_limbs, num_modulus, plain_modulus_bitlen_);

  // convert position form to RNS form
  auto pool = seal::MemoryManager::GetPool();
  const auto *rns_tool = cntxt_dat->rns_tool();
  rns_tool->base_q()->decompose(Q_div_plain.data(), pool);
  Q_div_plain_mod_qi_.resize(num_modulus);
  for (size_t i = 0; i < num_modulus; ++i) {
    Q_div_plain_mod_qi_[i].set(Q_div_plain[i], coeff_modulus[i]);
  }

  const auto &base_Q = *cntxt_dat->rns_tool()->base_q();
  base_Q_to_gamma_conv_ = std::make_unique<seal::util::BaseConverter>(
      base_Q, seal::util::RNSBase({gamma_}, pool), pool);

  // Q/qi mod t
  punctured_base_mod_plain_.resize(num_modulus);
  for (size_t l = 0; l < num_modulus; ++l) {
    const uint64_t *Q_qi_limbs =
        base_Q.punctured_prod_array() + l * num_modulus;
    punctured_base_mod_plain_[l] =
        limbs_mod_2k(Q_qi_limbs, num_modulus, plain_modulus_bitlen_);
  }

  // -Q^{-1} mod t
  // gamma^{-1} mod t
  neg_inv_Q_mod_plain_ = (-inv_mod_2k(base_Q.base_prod()[0])) & mod_plain_mask_;
  inv_gamma_mod_plain_ = inv_mod_2k(gamma_.value()) & mod_plain_mask_;

  // -Q^{-1} mod gamma
  neg_inv_Q_mod_gamma_ = [&]() {
    using namespace seal::util;
    auto Q_mod_gamma = modulo_uint(base_Q.base_prod(), num_modulus, gamma_);
    u64 inv;
    if (not try_invert_uint_mod(Q_mod_gamma, gamma_, inv)) {
      throw std::invalid_argument("Q_mod_gamma^{-1}");
    }
    MultiplyUIntModOperand ret;
    ret.set(negate_uint_mod(inv, gamma_), gamma_);
    return ret;
  }();

  // gamma*t mod Q
  gamma_plain_mod_Q_.resize(num_modulus);
  uint128_t t0 = static_cast<u128>(1) << (plain_modulus_bitlen_ / 2);
  uint128_t t1 = static_cast<u128>(1)
                 << (plain_modulus_bitlen_ - plain_modulus_bitlen_ / 2);

  std::transform(coeff_modulus.begin(), coeff_modulus.end(),
                 gamma_plain_mod_Q_.data(), [&](const seal::Modulus &prime) {
                   // 2^k0 mod prime
                   u64 t = barrett_reduce(t0, prime);
                   // 2^k0 mod prime * 2^k1 mod prime -> 2^k mod prime
                   t = seal::util::multiply_uint_mod(
                       t, barrett_reduce(t1, prime), prime);

                   seal::util::MultiplyUIntModOperand ret;
                   u64 g = seal::util::barrett_reduce_64(gamma_.value(), prime);
                   ret.set(seal::util::multiply_uint_mod(g, t, prime), prime);
                   return ret;
                 });
}

void BFVModuliScaler::scale_up_at(gsl::span<const u64> input,
                                  std::size_t modulus_index,
                                  gsl::span<u64> output) const {
  if (modulus_index >= coeff_modulus_size()) {
    throw std::invalid_argument(std::string(__func__) +
                                ": modulus_index out-of-bound");
  }
  if (output.size() != input.size()) {
    throw std::invalid_argument(std::string(__func__) +
                                ": input.size mismatch");
  }

  const auto &modulus = context_.key_context_data()->parms().coeff_modulus();

  // round(Q/t*x) = k*x + round(r*x/t) where k = floor(Q/t), r = Q mod t
  // round(Q/t*x) mod qi = ((k mod qi)*x + round(r*x/t)) mod qi
  std::transform(input.begin(), input.end(), output.data(), [&](u64 x) {
    u64 x64 = seal::util::barrett_reduce_64(x, modulus[modulus_index]);
    uint64_t u = seal::util::multiply_uint_mod(
        x64, Q_div_plain_mod_qi_[modulus_index], modulus[modulus_index]);
    u64 v = ((Q_mod_plain_ * x + plain_half_) >> plain_modulus_bitlen_);

    return seal::util::barrett_reduce_64(u + v, modulus[modulus_index]);
  });
}

void BFVModuliScaler::scale_up(gsl::span<const u64> input,
                               gsl::span<u64> output) const {
  std::size_t num_coeff = input.size();
  std::size_t num_modulus = coeff_modulus_size();
  if (output.size() != num_coeff * num_modulus) {
    throw std::invalid_argument(std::string(__func__) +
                                ": output.size mismatch");
  }

  for (std::size_t j = 0; j < num_modulus; ++j) {
    scale_up_at(input, j, output.subspan(j * num_coeff, num_coeff));
  }
}

void BFVModuliScaler::scale_down(gsl::span<const u64> input,
                                 gsl::span<u64> output) const {
  // Ref: Bajard et al. "A Full RNS Variant of FV like Somewhat Homomorphic
  // Encryption Schemes" (Section 3.2 & 3.3)
  namespace su = seal::util;

  std::size_t num_coeff = output.size();
  std::size_t num_modulus = coeff_modulus_size();

  if (input.size() % num_modulus != 0) {
    throw std::invalid_argument(std::string(__func__) + ": invalid input.size");
  }

  if (input.size() != num_coeff * num_modulus) {
    throw std::invalid_argument(std::string(__func__) +
                                ": output.size mismatch");
  }

  auto cntxt = context_.key_context_data();
  const auto &base_Q = *cntxt->rns_tool()->base_q();
  const auto &modulus = cntxt->parms().coeff_modulus();
  auto pool = seal::MemoryManager::GetPool();
  auto tmp = su::allocate_uint(input.size(), pool);

  // 1. multiply with gamma*plain
  for (std::size_t j = 0; j < num_modulus; ++j) {
    const auto *src_ptr = input.data() + j * num_coeff;
    auto *dst_ptr = tmp.get() + j * num_coeff;
    multiply_poly_scalar_coeffmod(src_ptr, num_coeff, gamma_plain_mod_Q_[j],
                                  modulus[j], dst_ptr);
  }

  // 2-1 FastBase convert from baseQ to {gamma}
  auto base_on_gamma = su::allocate_uint(num_coeff, pool);
  su::ConstRNSIter inp(tmp.get(), num_coeff);
  su::RNSIter outp(base_on_gamma.get(), num_coeff);
  base_Q_to_gamma_conv_->fast_convert_array(inp, outp, pool);
  // 2-2 Then multiply with -Q^{-1} mod gamma
  multiply_poly_scalar_coeffmod(base_on_gamma.get(), num_coeff,
                                neg_inv_Q_mod_gamma_, gamma_,
                                base_on_gamma.get());

  // 3-1 FastBase convert from baseQ to {plain}
  // NOTE: overwrite the `tmp` (tmp is gamma*plain*x mod Q)
  const auto *inv_punctured = base_Q.inv_punctured_prod_mod_base_array();
  for (std::size_t j = 0; j < num_modulus; ++j) {
    auto *src_ptr = tmp.get() + j * num_coeff;
    su::multiply_poly_scalar_coeffmod(src_ptr, num_coeff, inv_punctured[j],
                                      modulus[j], src_ptr);
  }
  // sum_i (x * (Q/qi)^{-1} mod qi) * (Q/qi) mod t
  std::vector<u64> base_on_plain(num_coeff);
  std::fill_n(base_on_plain.data(), num_coeff, 0);
  for (std::size_t j = 0; j < num_modulus; ++j) {
    const u64 factor = punctured_base_mod_plain_[j];
    auto *ptr = tmp.get() + j * num_coeff;
    std::transform(
        ptr, ptr + num_coeff, base_on_plain.data(), base_on_plain.data(),
        [factor](u64 x, u64 y) { return y + static_cast<u64>(x) * factor; });
  }

  // 3-2 Then multiply with -Q^{-1} mod t
  for (auto &x : base_on_plain) {
    x = (x * neg_inv_Q_mod_plain_) & mod_plain_mask_;
  }

  // 4 Correct sign: (base_on_t - [base_on_gamma]_gamma) * gamma^{-1} mod t
  // NOTE(juhou):
  // `base_on_gamma` and `base_on_t` together gives
  // `gamma*(x + t*r) + round(gamma*v/q) - e` mod gamma*t for some unknown v and
  // e. (Key point): Taking `base_on_gamma` along equals to
  //    `round(gamma*v/q) - e mod gamma`
  // When gamma > v, e, we can have the centered remainder
  // [round(gamma*v/q) - e mod gamma]_gamma = round(gamma*v/q) - e.
  // As a result, `base_on_t - [base_on_gamma]_gamma mod t` will cancel out the
  // last term and gives `gamma*(x + t*r) mod t`.
  // Finally, multiply with `gamma^{-1} mod t` gives `x mod t`.
  uint64_t gamma_div_2 = gamma_.value() >> 1;
  std::transform(base_on_gamma.get(), base_on_gamma.get() + num_coeff,
                 base_on_plain.data(), output.data(),
                 [&](u64 on_gamma, u64 on_plain) {
                   // [0, gamma) -> [-gamma/2, gamma/2]
                   if (on_gamma > gamma_div_2) {
                     return ((on_plain + gamma_.value() - on_gamma) *
                             inv_gamma_mod_plain_) &
                            mod_plain_mask_;
                   } else {
                     return ((on_plain - on_gamma) * inv_gamma_mod_plain_) &
                            mod_plain_mask_;
                   }
                 });
}
