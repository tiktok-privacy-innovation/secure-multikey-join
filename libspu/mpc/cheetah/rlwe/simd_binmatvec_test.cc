#include <algorithm>
#include <mutex>
#include <random>

#include "absl/types/span.h"
#include "seal/seal.h"
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

class SparseBinaryMat {
 public:
  SparseBinaryMat(std::size_t dim_in, std::size_t dim_out);

  std::size_t dim_in() const { return dim_in_; }

  std::size_t dim_out() const { return dim_out_; }

  void set(std::size_t src_indx, std::size_t out_indx);

  std::vector<std::uint64_t> get_diag(std::size_t diag_indx) const {
    std::vector<std::uint64_t> diag(diag_len());
    get_diag(diag_indx, absl::MakeSpan(diag));
    return diag;
  }

  void get_diag(std::size_t diag_indx, absl::Span<std::uint64_t> out) const;

  std::size_t num_diags() const { return num_diags_; }

  std::size_t diag_len() const { return diag_len_; }

  std::size_t count_nonzero_diags() const { return diags_.size(); }

  void compute_matvec(absl::Span<const std::uint64_t> input,
                      absl::Span<std::uint64_t> output) const;

  bool is_empty_diagnoal(std::size_t diag_indx) const {
    ENFORCE_ARG(diag_indx < num_diags(), "diag_indx");
    return diags_.find(diag_indx) == diags_.end();
  }

 private:
  std::size_t dim_in_;
  std::size_t dim_out_;
  std::size_t dim_in_aligned_;   // bit_ceil(dim_in)
  std::size_t dim_out_aligned_;  // bit_ceil(dim_out)

  std::size_t num_diags_;  // min(dim_in_aligned_, dim_out_aligned_)
  std::size_t diag_len_;   // max(dim_in_aligned_, dim_out_aligned_)

  using Set = std::unordered_set<std::size_t>;
  std::unordered_map<size_t, Set> diags_;
};

SparseBinaryMat::SparseBinaryMat(std::size_t dim_in, std::size_t dim_out)
    : dim_in_(dim_in), dim_out_(dim_out) {
  ENFORCE_ARG(dim_in_ > 0, "dim_in");
  ENFORCE_ARG(dim_out_ > 0, "dim_out");
  dim_in_aligned_ = absl::bit_ceil(dim_in_);
  dim_out_aligned_ = absl::bit_ceil(dim_out_);
  num_diags_ = std::min(dim_in_aligned_, dim_out_aligned_);
  diag_len_ = std::max(dim_in_aligned_, dim_out_aligned_);
}

void SparseBinaryMat::set(std::size_t src_indx, std::size_t out_indx) {
  ENFORCE_ARG(src_indx < dim_in_, "src_index");
  ENFORCE_ARG(out_indx < dim_out_, "out_index");
  // take mod num_diags by masking since num_diags is two-power number
  auto diag_indx = (src_indx + num_diags_ - out_indx) & (num_diags_ - 1);
  auto kv = diags_.find(diag_indx);

  std::size_t idx = num_diags_ * (src_indx / num_diags_) + out_indx;

  if (kv == diags_.end()) {
    Set set;
    set.insert(idx);
    diags_.insert({diag_indx, set});
  } else {
    kv->second.insert(idx);
  }
}

void SparseBinaryMat::get_diag(std::size_t diag_indx,
                               absl::Span<std::uint64_t> out) const {
  ENFORCE_ARG(diag_indx < num_diags(), "diag_indx");
  ENFORCE_ARG(out.size() == diag_len(), "out.size");
  std::fill_n(out.data(), out.size(), 0);
  auto kv = diags_.find(diag_indx);
  if (kv == diags_.end()) {
    return;
  }

  for (auto out_indx : kv->second) {
    out[out_indx] = 1;
  }
}

void SparseBinaryMat::compute_matvec(absl::Span<const uint64_t> input,
                                     absl::Span<uint64_t> output) const {
  ENFORCE_ARG(dim_in() == input.size(), "input.size()");
  ENFORCE_ARG(dim_out() == output.size(), "output.size()");

  auto right_rot_vec_inplace = [](absl::Span<uint64_t> v, size_t steps) {
    steps %= v.size();
    if (steps == 0) {
      return;
    }
    std::rotate(v.begin(), v.begin() + steps, v.end());
  };

  auto add_vec_inplace = [](absl::Span<uint64_t> v,
                            absl::Span<const uint64_t> u) {
    ENFORCE_ARG(v.size() <= u.size(), "add_vec");
    std::transform(v.cbegin(), v.cend(), u.cbegin(), v.data(),
                   std::plus<uint64_t>());
  };

  auto mul_vec = [](absl::Span<const uint64_t> v,
                    absl::Span<const uint64_t> u) {
    ENFORCE_ARG(v.size() == u.size(), "mul_vec");
    std::vector<uint64_t> res(v.size());
    std::transform(v.cbegin(), v.cend(), u.cbegin(), res.data(),
                   std::multiplies<uint64_t>());
    return res;
  };

  std::vector<uint64_t> vec(input.begin(), input.end());

  std::vector<uint64_t> result = mul_vec(input, get_diag(0));
  std::copy_n(result.data(), result.size(), output.data());

  for (size_t i = 1; i < num_diags(); ++i) {
    right_rot_vec_inplace(absl::MakeSpan(vec), 1);
    add_vec_inplace(output, mul_vec(vec, get_diag(i)));
  }

  std::size_t cext = absl::bit_ceil(dim_in());
  std::size_t rext = absl::bit_ceil(dim_out());

  // std::size_t half_slots = num_slots() >> 1;
  for (std::size_t rot = rext; rot < cext; rot <<= 1) {
    std::vector<uint64_t> copy(output.begin(), output.end());
    right_rot_vec_inplace(absl::MakeSpan(copy), rot);
    add_vec_inplace(output, copy);
  }
}

class SparseSquareBinaryMat {
 public:
  SparseSquareBinaryMat(std::size_t ndim);

  std::size_t ndim() const { return ndim_; }

  void set(std::size_t src_indx, std::size_t out_indx);

  std::vector<std::uint64_t> get_diag(std::size_t diag_indx) const {
    std::vector<std::uint64_t> diag(ndim_);
    get_diag(diag_indx, absl::MakeSpan(diag));
    return diag;
  }

  void get_diag(std::size_t diag_indx, absl::Span<std::uint64_t> out) const;

  std::size_t num_diags() const { return ndim_; }

  std::size_t count_nonzero_diags() const { return diags_.size(); }

  bool is_empty_diagnoal(std::size_t diag_indx) const {
    ENFORCE_ARG(diag_indx < num_diags(), "diag_indx");
    return diags_.find(diag_indx) == diags_.end();
  }

  void compute_matvec(absl::Span<const uint64_t> input,
                      absl::Span<uint64_t> output) const;

  void compute_matvec(const seal::Ciphertext &input,
                      const seal::GaloisKeys &gkeys,
                      const seal::SEALContext &context,
                      seal::Ciphertext &out) const;

 private:
  std::size_t ndim_;

  using Set = std::unordered_set<std::size_t>;
  std::unordered_map<size_t, Set> diags_;
};

SparseSquareBinaryMat::SparseSquareBinaryMat(std::size_t ndim) : ndim_(ndim) {
  ENFORCE_ARG(ndim > 0, "ndim");
}

void SparseSquareBinaryMat::set(std::size_t src_indx, std::size_t out_indx) {
  ENFORCE_ARG(src_indx < ndim_, "src_index >= ndim");
  ENFORCE_ARG(out_indx < ndim_, "out_index >= ndim");
  std::size_t diag_indx = (src_indx + ndim_ - out_indx) % ndim_;
  auto kv = diags_.find(diag_indx);
  if (kv == diags_.end()) {
    Set set;
    set.insert(out_indx);
    diags_.insert({diag_indx, set});
  } else {
    kv->second.insert(out_indx);
  }
}

void SparseSquareBinaryMat::get_diag(std::size_t diag_indx,
                                     absl::Span<std::uint64_t> out) const {
  ENFORCE_ARG(diag_indx < num_diags(), "diag_indx");
  ENFORCE_ARG(out.size() == ndim_, "out.size");
  std::fill_n(out.data(), out.size(), 0);
  auto kv = diags_.find(diag_indx);
  if (kv == diags_.end()) {
    return;
  }
  for (auto out_indx : kv->second) {
    out[out_indx] = 1;
  }
}

void SparseSquareBinaryMat::compute_matvec(absl::Span<const uint64_t> input,
                                           absl::Span<uint64_t> output) const {
  ENFORCE_ARG(ndim() == input.size(), "input.size()");
  ENFORCE_ARG(ndim() == output.size(), "output.size()");

  auto right_rot_vec_inplace = [](absl::Span<uint64_t> v, size_t steps) {
    steps %= v.size();
    if (steps == 0) {
      return;
    }
    std::rotate(v.begin(), v.begin() + steps, v.end());
  };

  auto add_vec_inplace = [](absl::Span<uint64_t> v,
                            absl::Span<const uint64_t> u) {
    std::transform(v.cbegin(), v.cend(), u.cbegin(), v.data(),
                   std::plus<uint64_t>());
  };

  auto mul_vec = [](absl::Span<const uint64_t> v,
                    absl::Span<const uint64_t> u) {
    std::vector<uint64_t> res(v.size());
    std::transform(v.cbegin(), v.cend(), u.cbegin(), res.data(),
                   std::multiplies<uint64_t>());
    return res;
  };

  std::vector<uint64_t> vec(input.begin(), input.end());

  std::vector<uint64_t> result = mul_vec(input, get_diag(0));
  std::copy_n(result.data(), result.size(), output.data());

  for (size_t i = 1; i < num_diags(); ++i) {
    right_rot_vec_inplace(absl::MakeSpan(vec), 1);
    add_vec_inplace(output, mul_vec(vec, get_diag(i)));
  }
}

void compute_matvec_single(const SparseBinaryMat &mat,
                           const seal::Ciphertext &input,
                           const seal::GaloisKeys &gkeys,
                           const seal::SEALContext &context,
                           seal::Ciphertext &out) {
  std::size_t poly_N = input.poly_modulus_degree();
  ENFORCE_ARG(mat.diag_len() <= poly_N, "diagnoal len out-of-bound");
  ENFORCE_ARG(mat.num_diags() * 2 <= poly_N, "num_diags out-of-bound");

  std::size_t num_diag = mat.num_diags();
  std::size_t giant_step = std::ceil(std::sqrt(1. * num_diag));
  std::size_t baby_step = (num_diag + giant_step - 1) / giant_step;

  std::vector<seal::Ciphertext> rotated_input(giant_step);
  seal::Evaluator evaluator(context);

  rotated_input[0] = input;
  if (not input.is_ntt_form()) {
    evaluator.transform_to_ntt_inplace(rotated_input[0]);
  }

  yacl::parallel_for(1, giant_step, 1, [&](std::size_t bgn, std::size_t end) {
    for (std::size_t j = bgn; j < end; ++j) {
      evaluator.rotate_rows(rotated_input[0], j, gkeys, rotated_input[j]);
    }
  });

  seal::BatchEncoder encoder(context);

  out.release();
  std::mutex result_guard;

  std::atomic<int> nnz = 0;
  yacl::parallel_for(0, baby_step, 1, [&](size_t kbgn, size_t kend) {
    std::vector<std::uint64_t> diag_vector(poly_N);
    seal::Plaintext encoded_diag;
    for (std::size_t k = kbgn; k < kend && k * giant_step < num_diag; ++k) {
      const size_t rhs_rot = k * giant_step;
      seal::Ciphertext inner_accum;

      for (std::size_t j = 0; j < giant_step; ++j) {
        const auto &rot_vec = rotated_input[j];
        std::size_t diag_indx = rhs_rot + j;
        if (diag_indx >= num_diag) break;

        // some diagnoals might be zero.
        if (mat.is_empty_diagnoal(diag_indx)) {
          continue;
        }
        nnz += 1;

        mat.get_diag(diag_indx, {diag_vector.data(), mat.diag_len()});

        if (mat.diag_len() == poly_N) {
          std::size_t half_N = poly_N / 2;
          // right rotate vec[0: poly_N/2)
          std::rotate(diag_vector.begin(),
                      diag_vector.begin() + half_N - rhs_rot,
                      diag_vector.begin() + half_N);
          // right rotate vec[poly_N/2, N)
          std::rotate(diag_vector.begin() + half_N,
                      diag_vector.begin() + poly_N - rhs_rot,
                      diag_vector.begin() + poly_N);
        } else {
          // 2 * num_dim < poly_N
          std::rotate(diag_vector.begin(), diag_vector.begin() + rhs_rot,
                      diag_vector.end());
        }
        // zero padding
        std::size_t aligned_len = absl::bit_ceil(mat.diag_len());
        std::fill_n(diag_vector.data() + mat.diag_len(),
                    aligned_len - mat.diag_len(), 0);
        // self repetition
        for (std::size_t j = 1; j < poly_N / aligned_len; ++j) {
          std::copy_n(diag_vector.cbegin(), aligned_len,
                      diag_vector.begin() + j * aligned_len);
        }

        encoder.encode(diag_vector, encoded_diag);

        if (inner_accum.size() > 0) {
          seal::Ciphertext mul;
          evaluator.multiply_plain(rot_vec, encoded_diag, mul);
          evaluator.add_inplace(inner_accum, mul);
        } else {
          evaluator.multiply_plain(rot_vec, encoded_diag, inner_accum);
        }
      }  // inner-loop

      if (inner_accum.size() == 0) {
        continue;
      }

      // Giant-step rotations: 11% time
      evaluator.rotate_rows_inplace(inner_accum, rhs_rot, gkeys);

      // race-write
      std::lock_guard<std::mutex> lock(result_guard);
      if (out.size() > 0) {
        evaluator.add_inplace(out, inner_accum);
      } else {
        out = inner_accum;
      }
    }
  });  // parallel_for

  std::size_t cext = absl::bit_ceil(mat.dim_in());
  std::size_t rext = absl::bit_ceil(mat.dim_out());
  for (std::size_t rot = rext; rot < cext; rot <<= 1) {
    seal::Ciphertext cpy = out;
    if (rot * 2 == poly_N) {
      evaluator.rotate_columns_inplace(cpy, gkeys);
    } else {
      evaluator.rotate_rows_inplace(cpy, rot, gkeys);
    }
    evaluator.add_inplace(out, cpy);
  }
}

void compute_matvec_optimized(const SparseSquareBinaryMat &mat0,
                              const SparseSquareBinaryMat &mat1,
                              const seal::Ciphertext &input,
                              const seal::GaloisKeys &gkeys,
                              const seal::SEALContext &context,
                              seal::Ciphertext &out) {
  std::size_t poly_N = input.poly_modulus_degree();
  ENFORCE_ARG(mat0.ndim() == mat1.ndim(), "matrices ndim mismatch");
  ENFORCE_ARG(mat0.ndim() * 2 <= poly_N, "ndim, poly_deg mismatch");

  std::size_t num_dim = mat0.ndim();
  std::size_t num_diag = num_dim;
  std::size_t giant_step = std::ceil(std::sqrt(1. * num_diag));
  std::size_t baby_step = (num_diag + giant_step - 1) / giant_step;

  std::vector<seal::Ciphertext> rotated_input(giant_step);
  seal::Evaluator evaluator(context);

  rotated_input[0] = input;
  if (not input.is_ntt_form()) {
    evaluator.transform_to_ntt_inplace(rotated_input[0]);
  }

  yacl::ElapsedTimer _timer;
  yacl::parallel_for(1, giant_step, 1, [&](std::size_t bgn, std::size_t end) {
    for (std::size_t j = bgn; j < end; ++j) {
      evaluator.rotate_rows(rotated_input[0], j, gkeys, rotated_input[j]);
    }
  });
  printf("Gstep %fms\n", _timer.CountMs());
  _timer.Restart();

  seal::BatchEncoder encoder(context);
  out.release();
  std::mutex result_guard;
  std::atomic<int> nnz = 0;
  yacl::parallel_for(0, baby_step, 1, [&](size_t kbgn, size_t kend) {
    std::vector<std::uint64_t> diag_vector(poly_N);
    seal::Plaintext encoded_diag;
    for (std::size_t k = kbgn; k < kend && k * giant_step < num_diag; ++k) {
      const size_t rhs_rot = k * giant_step;
      seal::Ciphertext inner_accum;

      for (std::size_t j = 0; j < giant_step; ++j) {
        const auto &rot_vec = rotated_input[j];
        std::size_t diag_indx = rhs_rot + j;
        if (diag_indx >= num_diag) break;

        // some diagnoals might be zero.
        if (mat0.is_empty_diagnoal(diag_indx) and
            mat1.is_empty_diagnoal(diag_indx)) {
          continue;
        }
        nnz += 1;

        mat0.get_diag(diag_indx, {diag_vector.data(), num_dim});
        mat1.get_diag(diag_indx, {diag_vector.data() + num_dim, num_dim});

        if (2 * num_dim == poly_N) {
          std::size_t half_N = poly_N / 2;
          // right rotate vec[0: poly_N/2)
          std::rotate(diag_vector.begin(),
                      diag_vector.begin() + half_N - rhs_rot,
                      diag_vector.begin() + half_N);
          // right rotate vec[poly_N/2, N)
          std::rotate(diag_vector.begin() + half_N,
                      diag_vector.begin() + poly_N - rhs_rot,
                      diag_vector.begin() + poly_N);
        } else {
          // 2 * num_dim < poly_N
          std::rotate(diag_vector.begin(), diag_vector.begin() + rhs_rot,
                      diag_vector.end());
        }

        std::size_t aligned_num_dim = absl::bit_ceil(2 * num_dim);
        std::fill_n(diag_vector.data() + 2 * num_dim,
                    aligned_num_dim - 2 * num_dim, 0);

        // self repetition
        if (aligned_num_dim < poly_N) {
          std::size_t num_repeat = poly_N / aligned_num_dim;
          for (std::size_t j = 1; j < num_repeat; ++j) {
            std::copy_n(diag_vector.cbegin(), aligned_num_dim,
                        diag_vector.begin() + j * aligned_num_dim);
          }
        }

        encoder.encode(diag_vector, encoded_diag);

        if (inner_accum.size() > 0) {
          seal::Ciphertext mul;
          evaluator.multiply_plain(rot_vec, encoded_diag, mul);
          evaluator.add_inplace(inner_accum, mul);
        } else {
          evaluator.multiply_plain(rot_vec, encoded_diag, inner_accum);
        }
      }  // inner-loop

      if (inner_accum.size() == 0) {
        continue;
      }

      // Giant-step rotations: 11% time
      // spu::mpc::cheetah::ModulusSwtichInplace(
      //     inner_accum, inner_accum.coeff_modulus_size() - 1, context);
      evaluator.rotate_rows_inplace(inner_accum, rhs_rot, gkeys);

      // race-write
      std::lock_guard<std::mutex> lock(result_guard);
      if (out.size() > 0) {
        evaluator.add_inplace(out, inner_accum);
      } else {
        out = inner_accum;
      }
    }
  });  // parallel_for

  printf("BStep %fms with %d nnz\n", _timer.CountMs(), nnz.load());
  _timer.Restart();

  std::size_t cext = absl::bit_ceil(2 * num_dim);
  std::size_t rext = absl::bit_ceil(num_dim);
  // std::size_t half_slots = num_slots() >> 1;
  for (std::size_t rot = rext; rot < cext; rot <<= 1) {
    seal::Ciphertext cpy = out;
    if (rot * 2 == poly_N) {
      evaluator.rotate_columns_inplace(cpy, gkeys);
    } else {
      evaluator.rotate_rows_inplace(cpy, rot, gkeys);
    }
    evaluator.add_inplace(out, cpy);
  }
  printf("Trace step %fms\n", _timer.CountMs());
}

class BlockedPermutationMatrix {
 public:
  BlockedPermutationMatrix(std::size_t block_size);

  // y[i] <- x[perm[i]]
  void set_permutation(absl::Span<const std::size_t> perm);

  // y[i] <- x[sel[i]]
  void set_selections(absl::Span<const std::size_t> sel, std::size_t dim_in);

  std::size_t dim_in() const { return dim_in_; }

  std::size_t dim_out() const { return dim_out_; }

  void permute(absl::Span<const uint64_t> input,
               absl::Span<uint64_t> output) const;

  std::size_t count_nonzero_diags() const {
    std::size_t count = 0;
    for (const auto &b : block_matrices_) {
      count += b.count_nonzero_diags();
    }
    return count;
  }

 protected:
  SparseSquareBinaryMat &get_block_matrix(std::size_t block_in,
                                          std::size_t block_out) {
    ENFORCE_ARG(block_in < num_in_blocks_, "(const) block_in");
    ENFORCE_ARG(block_out < num_out_blocks_, "(const) block_out");
    return block_matrices_.at(block_in * num_out_blocks_ + block_out);
  }

  const SparseSquareBinaryMat &get_block_matrix(std::size_t block_in,
                                                std::size_t block_out) const {
    ENFORCE_ARG(block_in < num_in_blocks_, "block_in");
    ENFORCE_ARG(block_out < num_out_blocks_, "block_out");
    return block_matrices_.at(block_in * num_out_blocks_ + block_out);
  }

 private:
  std::size_t block_size_ = 0;
  std::size_t dim_in_ = 0;
  std::size_t dim_out_ = 0;
  std::size_t num_in_blocks_ = 0;
  std::size_t num_out_blocks_ = 0;

  std::vector<SparseSquareBinaryMat> block_matrices_;
};

BlockedPermutationMatrix::BlockedPermutationMatrix(std::size_t block_size)
    : block_size_(block_size) {
  ENFORCE_ARG(block_size > 0, "block_size");
}

// y[i] <- x[perm[i]]
void BlockedPermutationMatrix::set_selections(absl::Span<const std::size_t> sel,
                                              std::size_t dim_in) {
  dim_out_ = sel.size();
  dim_in_ = dim_in;

  ENFORCE_ARG(dim_out_ > 0, "empty selection");
  ENFORCE_ARG(dim_in_ > 0, "dim_in");
  ENFORCE_ARG(std::all_of(sel.begin(), sel.end(),
                          [&](std::size_t idx) { return idx < dim_in_; }),
              "selection range");

  num_in_blocks_ = (dim_in_ + block_size_ - 1) / block_size_;
  num_out_blocks_ = (dim_out_ + block_size_ - 1) / block_size_;

  block_matrices_.resize(num_out_blocks_ * num_in_blocks_,
                         SparseSquareBinaryMat(block_size_));

  for (std::size_t dim_out = 0; dim_out < dim_out_; ++dim_out) {
    std::size_t dim_in = sel[dim_out];

    std::size_t block_in = dim_in / block_size_;
    std::size_t block_out = dim_out / block_size_;

    get_block_matrix(block_in, block_out)
        .set(dim_in % block_size_, dim_out % block_size_);
  }
}

void BlockedPermutationMatrix::set_permutation(
    absl::Span<const std::size_t> perm) {
  auto n = perm.size();
  ENFORCE_ARG(*std::min_element(perm.begin(), perm.end()) == 0,
              "permutation range (-)");
  ENFORCE_ARG(*std::max_element(perm.begin(), perm.end()) + 1 == n,
              "permutation range (+)");
  set_selections(perm, n);
}

void BlockedPermutationMatrix::permute(absl::Span<const uint64_t> input,
                                       absl::Span<uint64_t> output) const {
  ENFORCE_ARG(input.size() == dim_in_,
              "input.size got " + std::to_string(input.size()) + " expected " +
                  std::to_string(dim_in_));

  ENFORCE_ARG(output.size() == dim_out_, "output.size");

  std::fill_n(output.data(), output.size(), 0);
  std::vector<uint64_t> tmp_out(block_size_, 0);

  for (std::size_t block_out = 0; block_out < num_out_blocks_; ++block_out) {
    std::size_t dim_out_bgn = block_out * block_size_;
    std::size_t dim_out_end =
        std::min(dim_out_bgn + block_size_, output.size());
    std::size_t dim_out_len = dim_out_end - dim_out_bgn;
    auto out_subvec = output.subspan(dim_out_bgn, dim_out_len);

    for (std::size_t block_in = 0; block_in < num_in_blocks_; ++block_in) {
      std::size_t dim_in_bgn = block_in * block_size_;
      std::size_t dim_in_end = std::min(dim_in_bgn + block_size_, input.size());
      std::size_t dim_in_len = dim_in_end - dim_in_bgn;
      auto input_subvec = input.subspan(dim_in_bgn, dim_in_len);

      if (dim_in_len == block_size_) {
        get_block_matrix(block_in, block_out)
            .compute_matvec(input_subvec, absl::MakeSpan(tmp_out));
      } else {
        std::vector<uint64_t> tmp_vec(block_size_, 0);
        std::copy_n(input_subvec.data(), dim_in_len, tmp_vec.data());

        get_block_matrix(block_in, block_out)
            .compute_matvec(tmp_vec, absl::MakeSpan(tmp_out));
      }

      std::transform(out_subvec.begin(), out_subvec.end(), tmp_out.data(),
                     out_subvec.data(), std::plus<uint64_t>());
    }
  }
}

int test_SparseSquareBinaryMat() {
  size_t n = 123;
  std::vector<size_t> index(n);
  std::iota(index.begin(), index.end(), 0);

  std::vector<uint64_t> val(n);
  std::uniform_int_distribution<uint64_t> uniform(0, -1);
  std::default_random_engine rdv(std::time(0));
  std::generate_n(val.data(), val.size(), [&]() { return uniform(rdv); });
  std::sort(index.begin(), index.end(),
            [&](size_t i, size_t j) { return val[i] < val[j]; });

  SparseSquareBinaryMat mat(n);
  for (size_t i = 0; i < n; ++i) {
    mat.set(/*src*/ index[i], /*dst*/ i);
  }

  // rotate-dot mul-then sum
  std::vector<uint64_t> result(n);
  mat.compute_matvec(val, absl::MakeSpan(result));

  for (size_t i = 0; i < n; ++i) {
    if (result[i] != val[index[i]]) {
      printf("test_SparseSquareBinaryMat: failed\n");
      return -1;
    }
  }

  printf("test_SparseSquareBinaryMat: passed\n");
  return 0;
}

int test_PermutationMatrix() {
  size_t n = 213;
  size_t bz = 16;

  std::default_random_engine rdv;
  std::vector<size_t> index(n);

  std::iota(index.begin(), index.end(), 0);
  std::shuffle(index.begin(), index.end(), rdv);

  std::vector<uint64_t> val(n);
  std::iota(val.begin(), val.end(), 1);

  BlockedPermutationMatrix pm(bz);
  pm.set_permutation(index);

  std::vector<uint64_t> sorted_val(n);
  pm.permute(val, absl::MakeSpan(sorted_val));

  for (size_t i = 0; i < n; ++i) {
    if (sorted_val[i] != val[index[i]]) {
      printf("test_PermutationMatrix : failed\n");
      return -1;
    }
  }

  printf("test_PermutationMatrix : passed\n");

  {
    size_t n = 10000000;  // 1M
    size_t bz = 8192;
    size_t bk = (n + bz - 1) / bz;
    uint64_t seed = std::random_device()();
    std::default_random_engine rdv(seed);
    std::vector<size_t> index(n);

    std::iota(index.begin(), index.end(), 0);
    std::shuffle(index.begin(), index.end(), rdv);

    BlockedPermutationMatrix pm(bz);
    pm.set_permutation(index);

    printf("permute 10^%.3f\n", std::log10(n));
    printf("Permutation: zero diags ratio %.3f%%\n",
           100. * (1.0 - pm.count_nonzero_diags() /
                             static_cast<double>(bk * bk * bz)));
  }

  return 0;
}

int test_SelectionMatrix() {
  size_t n = 213;
  size_t m = 10;
  size_t bz = 16;

  std::default_random_engine rdv;
  std::vector<size_t> index(n);

  std::iota(index.begin(), index.end(), 0);
  std::shuffle(index.begin(), index.end(), rdv);

  std::vector<uint64_t> val(n);
  std::iota(val.begin(), val.end(), 1);

  BlockedPermutationMatrix pm(bz);
  // out[i] = x[sel[i]]
  pm.set_selections(absl::MakeSpan(index.data(), m), n);

  std::vector<uint64_t> selected_val(m);
  pm.permute(val, absl::MakeSpan(selected_val));

  for (size_t i = 0; i < m; ++i) {
    if (selected_val[i] != val[index[i]]) {
      printf("test_Selection: failed\n");
      return -1;
    }
  }
  printf("test_Selection: passed\n");

  {
    size_t n = 10000000;  // 1M
    size_t m = 100000;
    printf("select 10^%.3f from 10^%.3f\n", std::log10(m), std::log10(n));
    size_t bz = 8192;
    size_t bk0 = (n + bz - 1) / bz;
    size_t bk1 = (m + bz - 1) / bz;
    uint64_t seed = std::random_device()();
    std::default_random_engine rdv(seed);
    std::vector<size_t> index(n);

    std::iota(index.begin(), index.end(), 0);
    std::shuffle(index.begin(), index.end(), rdv);

    BlockedPermutationMatrix pm(bz);
    pm.set_selections({index.data(), m}, n);
    printf("Selection: zero diags ratio %.3f%%\n",
           100. * (1.0 - pm.count_nonzero_diags() /
                             static_cast<double>(bk0 * bk1 * bz)));
  }
  return 0;
}

int testing() {
  (void)test_SparseSquareBinaryMat();
  // (void)test_PermutationMatrix();
  // (void)test_SelectionMatrix();
  return 0;
}

std::vector<std::uint32_t> GenerateGaloisElementsForBSGS(
    size_t n, const seal::util::GaloisTool &gtool) {
  int giant_step = std::ceil(std::sqrt(1. * n));
  int baby_step = (n + giant_step - 1) / giant_step;

  std::vector<std::uint32_t> gelt;
  for (int i = 1; i < giant_step; ++i) {
    gelt.push_back(gtool.get_elt_from_step(i));
  }

  for (int i = 1; i < baby_step; ++i) {
    gelt.push_back(gtool.get_elt_from_step(i * giant_step));
  }
  gelt.push_back(gtool.get_elt_from_step(0));
  return gelt;
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

void generate_square_matrix(std::vector<std::vector<uint64_t>> &mat, size_t n,
                            size_t target_size = 1000000) {
  mat.resize(n);
  std::uniform_int_distribution<uint64_t> uniform(0, 2);
  std::uniform_real_distribution<double> runiform(0, 1.0);
  std::default_random_engine rdv;
  double sparisty = 1. / ((target_size + n - 1) / n);
  for (size_t i = 0; i < n; ++i) {
    mat[i].resize(n, 0);
    if (runiform(rdv) < sparisty) {
      std::generate_n(mat[i].data(), mat[i].size(),
                      [&]() { return uniform(rdv); });
    }
  }
}

void encode_diags(const std::vector<std::vector<uint64_t>> &diags,
                  const seal::BatchEncoder &encoder,
                  std::vector<seal::Plaintext> &polys) {
  size_t n = diags.size();
  polys.resize(n);
  yacl::ElapsedTimer timer;

  yacl::parallel_for(0, n, [&](size_t bgn, size_t end) {
    for (size_t i = bgn; i < end; ++i) {
      encoder.encode(diags[i], polys[i]);
    }
  });
  printf("encode_diags n = %zd took %fms\n", n, timer.CountMs());
}

void bsgs_matvec(const std::vector<seal::Plaintext> &ecd_mat,
                 const seal::Ciphertext &vec, const seal::SEALContext &context,
                 const seal::GaloisKeys &gk, seal::Ciphertext &out) {
  yacl::ElapsedTimer timer;
  std::size_t num_diags = ecd_mat.size() / 2;
  std::size_t giant_step = std::ceil(std::sqrt(1. * num_diags));
  std::size_t baby_step = (num_diags + giant_step - 1) / giant_step;

  std::vector<seal::Ciphertext> rotated_vec(giant_step);
  seal::Evaluator evaluator(context);

  rotated_vec[0] = vec;
  if (not vec.is_ntt_form()) {
    evaluator.transform_to_ntt_inplace(rotated_vec[0]);
  }
  yacl::parallel_for(1, giant_step, 1, [&](std::size_t bgn, std::size_t end) {
    for (std::size_t j = bgn; j < end; ++j) {
      evaluator.rotate_rows(rotated_vec[0], j, gk, rotated_vec[j]);
    }
  });

  out.release();
  std::mutex result_guard;

  timer.Restart();
  yacl::parallel_for(0, baby_step, 1, [&](size_t kbgn, size_t kend) {
    double time = 0;
    for (size_t k = kbgn; k < kend && k * giant_step < num_diags; ++k) {
      const size_t rhs_rot = k * giant_step;
      seal::Ciphertext inner_accum;

      // 85% time
      yacl::ElapsedTimer _timer;
      for (size_t j = 0; j < giant_step; ++j) {
        const auto &rot_vec = rotated_vec[j];
        size_t diag_idx = rhs_rot + j;
        if (diag_idx >= num_diags) break;
        // some diagnoals might be zero.
        if (ecd_mat[diag_idx].is_zero()) {
          continue;
        }

        if (inner_accum.size() > 0) {
          seal::Ciphertext mul;
          evaluator.multiply_plain(rot_vec, ecd_mat[diag_idx], mul);
          evaluator.add_inplace(inner_accum, mul);
        } else {
          evaluator.multiply_plain(rot_vec, ecd_mat[diag_idx], inner_accum);
        }
      }  // inner-loop
      time += _timer.CountMs();

      if (inner_accum.size() == 0) {
        continue;
      }

      // Giant-step rotations: 11% time
      // if (rhs_rot * 2 > num_diags) {
      //   evaluator.rotate_rows_inplace(
      //       inner_accum, -static_cast<int64_t>(num_diags - rhs_rot), gk);
      // } else {
      evaluator.rotate_rows_inplace(inner_accum, rhs_rot, gk);
      //}

      // race-write
      std::lock_guard<std::mutex> lock(result_guard);
      if (out.size() > 0) {
        evaluator.add_inplace(out, inner_accum);
      } else {
        out = inner_accum;
      }
    }  // outter-loop
    printf("gstep-rotate ook %fms\n", time);
  });

  printf("gstep ook %fms\n", timer.CountMs());
}

void count_zero_diag_on_selection(size_t m, size_t n, size_t N) {
  std::vector<size_t> selections(m);
  std::iota(selections.begin(), selections.end(), 0);

  uint64_t seed = std::random_device()();
  std::default_random_engine rdv(seed);

  for (size_t j = m; j < n; ++j) {
    std::uniform_int_distribution<size_t> uniform(0, m - 1);
    selections.at(uniform(rdv)) = j;
  }

  if (m < 32 and n < 32) {
    for (size_t i = 0; i < m; ++i) {
      if (i > 0 and (i % N) == 0) {
        printf("--------\n");
      }
      for (size_t j = 0; j < n; ++j) {
        if (j > 0 and (j % N) == 0) {
          printf("|");
        }
        if (selections[i] == j) {
          printf("*");
        } else {
          printf("0");
        }
      }
      printf("\n");
    }
  }

  size_t rblk = (m + N - 1) / N;
  size_t cblk = (n + N - 1) / N;
  size_t count = 0;

  for (size_t r = 0; r < rblk; ++r) {
    for (size_t c = 0; c < cblk; ++c) {
      const size_t row_bgn = r * N;
      const size_t row_end = std::min(row_bgn + N, m);
      const size_t col_bgn = c * N;
      const size_t col_end = std::min(col_bgn + N, n);

      std::vector<bool> counting(N, true);
      for (size_t ii = row_bgn; ii < row_end; ++ii) {
        size_t cc = selections[ii];
        if (cc < col_bgn or cc >= col_end) {
          continue;
        }
        size_t rr = ii % N;
        cc = cc % N;
        counting[rr >= cc ? rr - cc : cc - rr] = false;
      }

      for (bool c : counting) {
        count += static_cast<size_t>(c);
      }
    }
  }

  printf("zero sub-diag %zd. expected %f\n", count,
         count / static_cast<double>(rblk * cblk * N));
}

void count_zero_diag_on_perm(size_t n, size_t N) {
  std::vector<size_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  uint64_t seed = std::random_device()();
  std::default_random_engine rdv(seed);
  std::shuffle(idx.begin(), idx.end(), rdv);

  // for (size_t i = 0; i < n; ++i) {
  //   if (i > 0 and (i % N) == 0) {
  //     printf("--------\n");
  //   }
  //   for (size_t j = 0; j < n; ++j) {
  //     if (j > 0 and (j % N) == 0) {
  //       printf("|");
  //     }
  //     if (idx[i] == j) {
  //       printf("*");
  //     } else {
  //       printf("0");
  //     }
  //   }
  //   printf("\n");
  // }

  size_t blk = (n + N - 1) / N;
  size_t count = 0;
  for (size_t r = 0; r < blk; ++r) {
    for (size_t c = 0; c < blk; ++c) {
      const size_t row_bgn = r * N;
      const size_t row_end = std::min(row_bgn + N, n);
      const size_t col_bgn = c * N;
      const size_t col_end = std::min(col_bgn + N, n);

      std::vector<bool> counting(N, true);
      for (size_t ii = row_bgn; ii < row_end; ++ii) {
        size_t jj = idx[ii];
        if (jj < col_bgn or jj >= col_end) {
          continue;
        }
        counting[ii >= jj ? ii - jj : jj - ii] = false;
      }

      for (bool c : counting) {
        count += static_cast<size_t>(c);
      }
    }
  }

  printf("zero sub-diag %zd. expected %f\n", count,
         count / static_cast<double>(blk * blk * N));
}

// \sum_i diag[i] * lrot(vec, i)
//
//
// D = ceil(sqrt(n))
// \sum_g lrot(\sum_k rrot(diag[k]) * lrot(vec, k), D)
int main() {
  yacl::set_num_threads(8);
  size_t poly_deg = 8192;
  auto parms = DecideSEALParameters(poly_deg);

  seal::SEALContext context(parms, true, seal::sec_level_type::none);

  seal::KeyGenerator keygen(context);
  seal::SecretKey sk = keygen.secret_key();
  seal::PublicKey pk;
  keygen.create_public_key(pk);

  seal::GaloisKeys gk;
  keygen.create_galois_keys(
      GenerateGaloisElementsForBSGS(
          poly_N / 2, *context.first_context_data()->galois_tool()),
      gk);
  size_t dim_in = parms.poly_modulus_degree();
  size_t dim_out = dim_in / 2;

  std::vector<std::size_t> index(dim_in);
  std::iota(index.begin(), index.end(), 0);
  std::default_random_engine rdv;
  std::shuffle(index.begin(), index.end(), rdv);
  // std::fill_n(index.begin(), index.size(), 1);

  SparseBinaryMat binmat0(dim_in, dim_out);
  SparseBinaryMat binmat1(dim_in, dim_out);

  // (Nx2) * N x N
  // (Nx2) * N x N
  for (std::size_t i = 0; i < dim_out; ++i) {
    // y[i] <= x[index[i]]
    binmat0.set(index[i], i);
    binmat1.set(index[dim_out + i], i);
  }

  // (Nx2) * N => Nx2
  seal::Evaluator evaluator(context);
  seal::BatchEncoder encoder(context);

  std::vector<uint64_t> input(dim_in);
  std::iota(input.begin(), input.end(), 0);

  seal::Plaintext vec;
  encoder.encode(input, vec);

  seal::Encryptor encryptor(context, sk);
  seal::Ciphertext ct;
  encryptor.encrypt_symmetric(vec, ct);

  seal::Ciphertext out0;
  seal::Ciphertext out1;

  // a || a => a(X) || 0
  // b || b => b(X) || 0
  compute_matvec_single(binmat0, ct, gk, context, out0);
  compute_matvec_single(binmat1, ct, gk, context, out1);

  if (out0.is_ntt_form()) {
    evaluator.transform_from_ntt_inplace(out0);
    evaluator.transform_from_ntt_inplace(out1);
  }

  std::vector<uint64_t> output0(dim_in);
  std::vector<uint64_t> output1(dim_in);
  seal::Decryptor decryptor(context, sk);
  seal::Plaintext pt;
  decryptor.decrypt(out0, pt);

  encoder.decode(pt, output0);
  // a0, a1, a2, ... || a0, a1, a2, ...,
  for (size_t i = 0; i < pt.coeff_count(); i += dim_out) {
    for (size_t j = 0; j < dim_out; ++j) {
      printf("%llu ", output0[i + j]);
    }
    printf("\n");
  }
  printf("\n");

  decryptor.decrypt(out1, pt);
  encoder.decode(pt, output1);

  for (size_t i = 0; i < dim_in; ++i) {
    printf("%llu ", input[index[i]]);
  }
  printf("\n");

  for (size_t i = 0; i < dim_out; ++i) {
    printf("%llu ", output0[i]);
    SPU_ENFORCE_EQ(output0[i], output0[dim_out + i]);
  }
  for (size_t i = 0; i < dim_out; ++i) {
    printf("%llu ", output1[i]);
    SPU_ENFORCE_EQ(output1[i], output1[dim_out + i]);
  }

  printf("\n\n");
  printf("\n");

  return 0;
}
