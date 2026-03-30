#include <algorithm>
#include <cstdint>
#include <mutex>
#include <random>
#include <unordered_map>

#include "absl/types/span.h"
#include "seal/seal.h"
#include "seal/util/polyarithsmallmod.h"
#include "utils.h"
#include "yacl/utils/elapsed_timer.h"
#include "yacl/utils/parallel.h"

#define ENFORCE_ARG(stmt, msg)                                  \
  do {                                                          \
    if (!(stmt)) {                                              \
      throw std::invalid_argument(std::string(__func__) + ":" + \
                                  std::string(msg));            \
    }                                                           \
  } while (0)

namespace {

[[maybe_unused]] void divide_degree_inplace(seal::Ciphertext& out,
                                            const seal::SEALContext& context) {
  auto ct = context.get_context_data(out.parms_id());
  SPU_ENFORCE(ct != nullptr);
  const auto& modulus = ct->parms().coeff_modulus();
  std::size_t num_modulus = modulus.size();
  std::size_t num_coeff = out.poly_modulus_degree();
  // N^{-1} mod Q
  //
  // N^{-1} mod t
  std::uint64_t inv_deg_mod_t =
      ct->plain_ntt_tables()->inv_degree_modulo().operand;
  for (std::size_t k = 0; k < out.size(); ++k) {
    auto ct_ptr = out.data(k);
    for (std::size_t j = 0; j < num_modulus; ++j) {
      std::uint64_t inv_deg_mod_t_mod_qi =
          seal::util::barrett_reduce_64(inv_deg_mod_t, modulus[j]);
      seal::util::multiply_poly_scalar_coeffmod(
          ct_ptr, num_coeff, inv_deg_mod_t_mod_qi, modulus[j], ct_ptr);
      ct_ptr += num_coeff;
    }
  }
}

[[maybe_unused]] void add_rlwe_inplace(seal::Ciphertext& out,
                                       const seal::Ciphertext& oth,
                                       const seal::SEALContext& context) {
  if (0 == out.size()) {
    out = oth;
    return;
  }
  seal::Evaluator evaluator(context);
  evaluator.add_inplace(out, oth);
}

[[maybe_unused]] void apply_inv_galois_inplace(
    seal::Plaintext& poly, std::uint32_t galois_elt,
    const seal::SEALContext& context) {
  auto ct = context.key_context_data();

  size_t N = ct->parms().poly_modulus_degree();
  SPU_ENFORCE(poly.coeff_count() == N);
  std::vector<std::uint64_t> tmp(N);

  std::uint64_t inv_galois_elt;
  SPU_ENFORCE(
      seal::util::try_invert_uint_mod(galois_elt, 2 * N, inv_galois_elt));

  auto& plain_modulus = ct->parms().plain_modulus();
  auto gtool = ct->galois_tool();
  gtool->apply_galois(poly.data(), inv_galois_elt, plain_modulus, tmp.data());
  std::copy_n(tmp.data(), N, poly.data());
}

[[maybe_unused]] void apply_galois(const seal::Plaintext& poly,
                                   std::uint32_t galois_elt,
                                   const seal::SEALContext& context,
                                   seal::Plaintext& out) {
  auto ct = context.key_context_data();
  auto gtool = ct->galois_tool();
  auto& plain_modulus = ct->parms().plain_modulus();

  out = seal::Plaintext();
  out.resize(poly.coeff_count());

  gtool->apply_galois(poly.data(), galois_elt, plain_modulus, out.data());
}

[[maybe_unused]] void multiply_plain_inplace(seal::Plaintext& poly,
                                             const seal::Plaintext& oth,
                                             const seal::SEALContext& context) {
  ENFORCE_ARG(poly.coeff_count() == oth.coeff_count(), "coeff_count");

  auto ct = context.key_context_data();
  auto plain_ntt = ct->plain_ntt_tables();
  const auto& plain_modulus = ct->parms().plain_modulus();

  std::vector<uint64_t> tmp(oth.coeff_count());
  std::copy_n(oth.data(), tmp.size(), tmp.data());
  seal::util::ntt_negacyclic_harvey(poly.data(), *plain_ntt);
  seal::util::ntt_negacyclic_harvey(tmp.data(), *plain_ntt);

  seal::util::dyadic_product_coeffmod(
      poly.data(), tmp.data(), poly.coeff_count(), plain_modulus, poly.data());

  seal::util::inverse_ntt_negacyclic_harvey(poly.data(), *plain_ntt);
}

[[maybe_unused]] void inverse_degree_inplace(seal::Plaintext& poly,
                                             const seal::SEALContext& context) {
  auto ct = context.key_context_data();
  auto plain_ntt = ct->plain_ntt_tables();
  const auto& plain_modulus = ct->parms().plain_modulus();
  auto inv_n = plain_ntt->inv_degree_modulo();
  seal::util::multiply_poly_scalar_coeffmod(poly.data(), poly.coeff_count(),
                                            inv_n, plain_modulus, poly.data());
}

[[maybe_unused]] void add_plain_inplace(seal::Plaintext& poly,
                                        const seal::Plaintext& oth,
                                        const seal::SEALContext& context) {
  if (poly.coeff_count() == 0) {
    poly = oth;
    return;
  }
  ENFORCE_ARG(poly.coeff_count() == oth.coeff_count(), "coeff_count");

  auto ct = context.key_context_data();
  const auto& plain_modulus = ct->parms().plain_modulus();
  seal::util::add_poly_coeffmod(poly.data(), oth.data(), poly.coeff_count(),
                                plain_modulus, poly.data());
}

}  // namespace

class SparseBinaryMat {
 public:
  SparseBinaryMat(std::size_t dim_in, std::size_t dim_out);

  std::size_t dim_in() const { return dim_in_; }

  std::size_t dim_out() const { return dim_out_; }

  void set(std::size_t src_indx, std::size_t out_indx) {
    ENFORCE_ARG(src_indx < dim_in_, "src_indx");
    ENFORCE_ARG(out_indx < dim_out_, "out_indx");
    auto kv = columns_.find(src_indx);
    if (kv == columns_.end()) {
      Set new_insert;
      new_insert.insert(out_indx);
      columns_.insert({src_indx, new_insert});
    } else {
      kv->second.insert(out_indx);
    }
  }

  bool test(std::size_t src_indx, std::size_t out_indx) const {
    auto kv = columns_.find(src_indx);
    if (kv == columns_.end()) {
      return false;
    }
    return kv->second.count(out_indx) > 0;
  }

  struct Entry {
    std::int64_t src_indx;
    std::int64_t out_indx;
  };

  std::vector<Entry> get_all_nonzero_entries() const {
    std::vector<Entry> entries;
    for (const auto& col : columns_) {
      std::int64_t src_indx = col.first;
      for (std::int64_t out_indx : col.second) {
        entries.push_back({.src_indx = src_indx, .out_indx = out_indx});
      }
    }
    return entries;
  }

 private:
  std::size_t dim_in_;
  std::size_t dim_out_;

  std::vector<Entry> nzero_entries_;
  using Set = std::unordered_set<std::size_t>;
  std::unordered_map<size_t, Set> columns_;
};

SparseBinaryMat::SparseBinaryMat(std::size_t dim_in, std::size_t dim_out)
    : dim_in_(dim_in), dim_out_(dim_out) {
  ENFORCE_ARG(dim_in_ > 0, "dim_in");
  ENFORCE_ARG(dim_out_ > 0, "dim_out");
}

// m_k(X) = \sum_{i, j}M[i, j]X^{i - (2k+1) * j}
void encode_sparse_matrix(const std::vector<SparseBinaryMat::Entry>& entries,
                          std::int64_t factor, const seal::SEALContext& context,
                          seal::Plaintext& out) {
  auto ct = context.first_context_data();
  const std::uint64_t plain_modulus = ct->parms().plain_modulus().value();
  const std::size_t poly_N = ct->parms().poly_modulus_degree();
  ENFORCE_ARG(factor > 0 and factor & 1, "need positive odd factor");
  ENFORCE_ARG((factor / 2) < (int64_t)poly_N, "factor out-of-bound");
  // ENFORCE_ARG(mat.dim_in() == poly_N, "mat.dim_in");
  // ENFORCE_ARG(mat.dim_out() == poly_N, "mat.dim_out");

  out = seal::Plaintext();
  out.resize(poly_N);
  std::fill_n(out.data(), poly_N, 0);
  int64_t* out_ptr = reinterpret_cast<int64_t*>(out.data());
  int64_t N = poly_N;
  for (const auto& entry : entries) {
    std::int64_t exp = entry.out_indx - factor * entry.src_indx;
    std::int64_t value = 1;

    if (exp < 0) {
      int64_t multiples = (std::abs(exp) + N - 1) / N;
      exp = multiples * N + exp;
      if (multiples & 1) {
        // odd
        value = -1;
      }
    }

    ENFORCE_ARG(exp < N, "exp < N");
    out_ptr[exp] += value;
  }

  std::transform(out_ptr, out_ptr + poly_N, out_ptr,
                 [&](int64_t x) { return x >= 0 ? x : plain_modulus + x; });
}

struct GaloisHelper {
  GaloisHelper(seal::SEALContext const& context) {
    const auto ct = context.key_context_data();
    const auto gtool = ct->galois_tool();
    std::size_t N = ct->parms().poly_modulus_degree();
    std::size_t n = N >> 1;
    std::size_t bstep = std::ceil(std::sqrt(n));
    std::size_t gstep = (n + bstep - 1) / bstep;
    std::uint32_t generator = 5;  // SEAL use 3, thus we use 5 here

    step_to_gelt_map_.insert({0U, 1U});
    galois_elt_.push_back(1);
    galois_elt_.push_back(generator);

    for (std::size_t step = 1; step < n; ++step) {
      std::uint32_t element = gtool->get_elt_from_step(step);
      step_to_gelt_map_.insert({step, element});
      if (step < bstep) {
        galois_elt_.push_back(element);
      }
    }

    for (std::size_t step = 1; step < gstep; ++step) {
      std::size_t _step = step * bstep;
      std::uint32_t element = gtool->get_elt_from_step(_step);
      galois_elt_.push_back(element);

      element = (generator * element) & (2 * N - 1);  // mod 2*N
      galois_elt_.push_back(element);
    }
  }

  const std::vector<std::uint32_t>& galois_elt() const { return galois_elt_; }

  std::uint32_t get_elt_from_step(std::size_t step) const {
    auto kv = step_to_gelt_map_.find(step);
    if (kv == step_to_gelt_map_.end()) {
      SPU_THROW("step = {} is absent", step);
    }
    return kv->second;
  }

  std::unordered_map<std::size_t, std::uint32_t> step_to_gelt_map_;
  std::vector<std::uint32_t> galois_elt_;
};

void compute_matvec_bsgs(const std::vector<seal::Plaintext>& encoded_matrix,
                         const seal::Ciphertext& input,
                         const seal::GaloisKeys& galois_keys,
                         const seal::SEALContext& context,
                         seal::Ciphertext& output) {
  std::size_t N = input.poly_modulus_degree();
  SPU_ENFORCE(encoded_matrix.size() == N);
  const std::size_t n = N >> 1;
  const std::size_t bstep = std::ceil(std::sqrt(n));
  const std::size_t gstep = (n + bstep - 1) / bstep;

  GaloisHelper galois_helper(context);
  seal::Evaluator evaluator(context);

  yacl::ElapsedTimer timer;
  std::vector<seal::Ciphertext> rotated_input(bstep);
  evaluator.transform_to_ntt(input, rotated_input[0]);
  yacl::parallel_for(1, bstep, [&](std::size_t bgn, std::size_t end) {
    for (std::size_t g = bgn; g < end; ++g) {
      evaluator.apply_galois(rotated_input[0],
                             galois_helper.get_elt_from_step(g), galois_keys,
                             rotated_input[g]);
    }
  });
  printf("baby step took %fms\n", timer.CountMs());

  std::mutex lock;
  output.release();
  timer.Restart();

  printf("parallel gstep %zd\n", gstep);
  yacl::parallel_for(0, gstep, 1, [&](std::size_t bgn, std::size_t end) {
    for (std::size_t g = bgn; g < end; ++g) {
      std::size_t offset = g * bstep;
      std::uint32_t offset_elt_3 = galois_helper.get_elt_from_step(offset);
      std::uint32_t offset_elt_5 = (5 * offset_elt_3) & (2 * N - 1);

      seal::Ciphertext inner_gen3;
      seal::Ciphertext inner_gen5;
      for (std::size_t b = 0; b < bstep; ++b) {
        std::size_t step = offset + b;
        if (step >= n) {
          break;
        }
        seal::Ciphertext tmp_ct;
        seal::Plaintext tmp_pt;
        std::uint32_t galois_elt;

        galois_elt = galois_helper.get_elt_from_step(step);
        tmp_pt = encoded_matrix[galois_elt / 2];
        apply_inv_galois_inplace(tmp_pt, offset_elt_3, context);
        evaluator.multiply_plain(rotated_input[b], tmp_pt, tmp_ct);
        add_rlwe_inplace(inner_gen3, tmp_ct, context);

        galois_elt = (5 * galois_elt) & (2 * N - 1);
        tmp_pt = encoded_matrix[galois_elt / 2];
        apply_inv_galois_inplace(tmp_pt, offset_elt_5, context);
        evaluator.multiply_plain(rotated_input[b], tmp_pt, tmp_ct);
        add_rlwe_inplace(inner_gen5, tmp_ct, context);
      }  // baby step

      spu::mpc::cheetah::ModulusSwtichInplace(
          inner_gen3, inner_gen3.coeff_modulus_size() - 1, context);
      spu::mpc::cheetah::ModulusSwtichInplace(
          inner_gen5, inner_gen5.coeff_modulus_size() - 1, context);

      evaluator.apply_galois_inplace(inner_gen3, offset_elt_3, galois_keys);
      evaluator.apply_galois_inplace(inner_gen5, offset_elt_5, galois_keys);

      std::lock_guard<std::mutex> guard(lock);
      add_rlwe_inplace(output, inner_gen3, context);
      add_rlwe_inplace(output, inner_gen5, context);
    }
  });

  printf("gstep took %fms\n", timer.CountMs());
  if (not input.is_ntt_form()) {
    evaluator.transform_from_ntt_inplace(output);
  }
}

seal::EncryptionParameters DecideSEALParameters(size_t poly_deg) {
  auto scheme_type = seal::scheme_type::bfv;
  auto parms = seal::EncryptionParameters(scheme_type);

  std::vector<int> modulus_bits = {59, 59, 59, 32};

  parms.set_use_special_prime(true);
  parms.set_poly_modulus_degree(poly_deg);
  auto modulus = seal::CoeffModulus::Create(poly_deg, modulus_bits);
  parms.set_plain_modulus(modulus.back());
  modulus.pop_back();
  parms.set_coeff_modulus(modulus);
  return parms;
}

int main() {
  yacl::set_num_threads(8);
  size_t poly_deg = 8192;
  auto parms = DecideSEALParameters(poly_deg);

  seal::SEALContext context(parms, true, seal::sec_level_type::none);
  GaloisHelper galois_helper(context);
  seal::KeyGenerator keygen(context);
  auto seckey = keygen.secret_key();
  seal::GaloisKeys galois_keys;
  keygen.create_galois_keys(galois_helper.galois_elt(), galois_keys);

  seal::Plaintext input_pt;
  input_pt.resize(poly_deg);
  std::iota(input_pt.data(), input_pt.data() + poly_deg, 1);

  std::vector<size_t> permutation(poly_deg);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::shuffle(permutation.begin(), permutation.end(),
               std::default_random_engine());

  seal::Ciphertext input_ct;
  seal::Encryptor encryptor(context, seckey);
  encryptor.encrypt_symmetric(input_pt, input_ct);

  SparseBinaryMat mat(poly_deg, poly_deg);
  for (size_t i = 0; i < permutation.size(); ++i) {
    mat.set(permutation[i], i);
  }

  yacl::ElapsedTimer e2e_timer;

  seal::Ciphertext output_ct;
  divide_degree_inplace(input_ct, context);

  std::vector<seal::Plaintext> encoded_matrix(poly_deg);
  auto entries = mat.get_all_nonzero_entries();
  yacl::ElapsedTimer timer;

  yacl::parallel_for(0, poly_deg, [&](size_t bgn, size_t end) {
    for (size_t i = bgn; i < end; ++i) {
      encode_sparse_matrix(entries, static_cast<int64_t>(2 * i + 1), context,
                           encoded_matrix[i]);
    }
  });
  printf("encode took %fms\n", timer.CountMs());

  compute_matvec_bsgs(encoded_matrix, input_ct, galois_keys, context,
                      output_ct);
  printf("One %zd-permutation took %fms\n", permutation.size(),
         e2e_timer.CountMs());

  seal::Plaintext output_pt;
  seal::Decryptor decryptor(context, seckey);
  decryptor.decrypt(output_ct, output_pt);
  for (size_t i = 0; i < output_pt.coeff_count(); ++i) {
    if (i < 8) {
      printf("%llu %llu\n", output_pt[i], input_pt[permutation[i]]);
    }
    SPU_ENFORCE_EQ(output_pt[i], input_pt[permutation[i]]);
  }
  return 0;
}
