#pragma once

#include "seal/context.h"

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using u128 = uint128_t;

class BFVModuliScaler {
 public:
  BFVModuliScaler(const seal::SEALContext& context, u32 plain_modulus_bitlen);

  void scale_up(gsl::span<const u64> input, gsl::span<u64> output) const;

  void scale_up_at(gsl::span<const u64> input, std::size_t modulus_index,
                   gsl::span<u64> output) const;

  void scale_down(gsl::span<const u64> input, gsl::span<u64> output) const;

 private:
  std::size_t coeff_modulus_size() const { return Q_div_plain_mod_qi_.size(); }

  void initialize_mod_2k(u32 plain_modulus_bitlen);

  seal::SEALContext context_;

  u32 plain_modulus_bitlen_;  // plain modulus is 2^k
  u128 mod_plain_mask_;       // 2^k - 1
  u128 plain_half_;           // 2^{k - 1}
  u128 Q_mod_plain_;          // Q mod 2^k

  seal::Modulus gamma_;       // auxiliary
  u128 neg_inv_Q_mod_plain_;  // -Q^{-1} mod 2^k
  u128 inv_gamma_mod_plain_;  // gamma^{-1} mod 2^k
  //
  seal::util::MultiplyUIntModOperand neg_inv_Q_mod_gamma_;  // -Q^{-1} mod gamma
  std::vector<seal::util::MultiplyUIntModOperand> Q_div_plain_mod_qi_;
  std::vector<u128> punctured_base_mod_plain_;
  std::vector<seal::util::MultiplyUIntModOperand> gamma_plain_mod_Q_;
  std::unique_ptr<seal::util::BaseConverter> base_Q_to_gamma_conv_;
};
