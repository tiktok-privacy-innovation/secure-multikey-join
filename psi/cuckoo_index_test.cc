// Copyright 2022 Ant Group Co., Ltd.
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

#include "psi/cuckoo_index.h"

#include <algorithm>
#include <random>

#include "gtest/gtest.h"
#include "yacl/crypto/hash/hash_utils.h"
#include "yacl/crypto/rand/rand.h"

namespace psi {

std::string SampleRandomString(size_t length) {
  const std::string charset =
      "0123456789"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz";

  std::string result;
  result.reserve(length);

  for (size_t i = 0; i < length; ++i) {
    result += charset[std::rand() % charset.size()];
  }

  return result;
}

bool IsValidPermutation(absl::Span<const int32_t> perm) {
  int32_t n = perm.size();
  if (!std::all_of(perm.begin(), perm.end(),
                   [n](int32_t x) { return x >= 0 && x < n; })) {
    return false;
  }

  std::vector<int32_t> copy(perm.begin(), perm.end());
  std::sort(copy.begin(), copy.end());
  for (int32_t i = 1; i < n; ++i) {
    if (copy[i] == copy[i - 1]) {
      return false;
    }
  }
  return true;
}

class CuckooIndexTest : public testing::TestWithParam<CuckooIndex::Options> {};

TEST_P(CuckooIndexTest, Works) {
  const auto& param = GetParam();
  {
    // Insert all in one call.
    CuckooIndex cuckoo_index(param);

    auto inputs = yacl::crypto::RandVec<uint128_t>(param.num_input);

    ASSERT_NO_THROW(cuckoo_index.Insert(absl::MakeSpan(inputs)));
    ASSERT_NO_THROW(cuckoo_index.SanityCheck());

    EXPECT_EQ(cuckoo_index.bins().size(), param.NumBins());
    EXPECT_EQ(cuckoo_index.stash().size(), param.num_stash);
    EXPECT_EQ(cuckoo_index.hashes().size(), param.num_input);
  }
  {
    // Insert by batches.
    CuckooIndex cuckoo_index(param);
    auto inputs = yacl::crypto::RandVec<uint128_t>(param.num_input);

    constexpr size_t kChunkSize = 1024;
    for (size_t i = 0; i < inputs.size(); i += kChunkSize) {
      size_t chunk_size = std::min(kChunkSize, inputs.size() - i);
      absl::Span<const uint128_t> chunk(inputs.data() + i, chunk_size);
      ASSERT_NO_THROW(cuckoo_index.Insert(chunk));
    }

    ASSERT_NO_THROW(cuckoo_index.SanityCheck());
    EXPECT_EQ(cuckoo_index.bins().size(), param.NumBins());
    EXPECT_EQ(cuckoo_index.stash().size(), param.num_stash);
    EXPECT_EQ(cuckoo_index.hashes().size(), param.num_input);
  }
}

TEST_P(CuckooIndexTest, Permutation) {
  const auto& param = GetParam();
  if (param.num_stash > 0) {
    return;
  }
  // Insert all in one call.
  CuckooIndex cuckoo_index(param);

  std::vector<std::string> keys(param.num_input);
  std::generate_n(keys.data(), keys.size(),
                  []() { return SampleRandomString(20); });

  std::vector<uint128_t> inputs(param.num_input);
  std::transform(
      keys.begin(), keys.end(), inputs.data(),
      [](const std::string_view& s) { return yacl::crypto::Blake3_128(s); });

  cuckoo_index.Insert(inputs);
  CuckooIndexHelper cuckoo_helper(cuckoo_index, param.num_input);

  const auto& bin_to_sample_perm = cuckoo_helper.bin_to_sample_perm();
  ASSERT_TRUE(IsValidPermutation(bin_to_sample_perm));

  const auto& sample_to_bin_map = cuckoo_helper.sample_to_bin_map();

  std::vector<std::string> keys_in_bins(cuckoo_helper.num_bin());
  for (size_t i = 0; i < param.num_input; ++i) {
    keys_in_bins[sample_to_bin_map[i]] = keys[i];
  }

  std::vector<std::string> permuted_bins(cuckoo_helper.num_sample());
  for (int32_t i = 0; i < cuckoo_helper.num_sample(); ++i) {
    permuted_bins[i] = keys_in_bins[bin_to_sample_perm[i]];
  }
  for (int32_t i = 0; i < cuckoo_helper.num_sample(); ++i) {
    ASSERT_EQ(keys[i], permuted_bins[i]);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Works_Instances, CuckooIndexTest,
    testing::Values(
        CuckooIndex::Options{(1 << 16), 0, 3, 1.27},      // 3-way, 1.27, 0
        CuckooIndex::Options{(1 << 20), 0, 3, 1.27},      // 3-way, 1.27, 0
        CuckooIndex::Options{(1 << 16), 8, 3, 1.2},       // 3-way, 1.2, 8
        CuckooIndex::Options{(1 << 16) + 1, 4, 2, 2.4},   // 2-way, 2.4, 4
        CuckooIndex::Options{(1 << 16) - 1, 4, 2, 2.4},   // 2-way, 2.4, 4
        CuckooIndex::Options{(1 << 20) + 17, 4, 3, 1.2},  // 3-way, 1.2, 4
        CuckooIndex::Options{(1 << 0), 0, 3, 1.27}        // dummy
        ));

TEST(CuckooIndexTest, Bad_StashTooSmall) {
  CuckooIndex cuckoo_index(CuckooIndex::Options{1 << 16, 0, 3, 1.1});
  auto inputs = yacl::crypto::RandVec<uint128_t>(1 << 16);

  ASSERT_THROW(cuckoo_index.Insert(absl::MakeSpan(inputs)), yacl::Exception);
}

TEST(CuckooIndexTest, Bad_SmallScaleFactor) {
  CuckooIndex cuckoo_index(CuckooIndex::Options{1 << 16, 8, 3, 1.01});
  auto inputs = yacl::crypto::RandVec<uint128_t>(1 << 16);

  ASSERT_THROW(cuckoo_index.Insert(absl::MakeSpan(inputs)), yacl::Exception);
}

}  // namespace psi
