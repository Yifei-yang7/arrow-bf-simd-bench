//
// Created by Yifei Yang on 5/2/24.
//

#define ANKERL_NANOBENCH_IMPLEMENT

#include "hash.hpp"
#include <nanobench.h>
#include <fmt/format.h>
#include <cassert>

static constexpr int ITER = 3;

void makeInput(uint64_t* vals, uint64_t count) {
  std::random_device rd;
  std::mt19937 gen(rd());
  auto dist = std::uniform_int_distribution<uint64_t>(0, UINT64_MAX);
  for (uint64_t i = 0; i < count; ++i) {
    vals[i] = dist(gen);
  }
}

void runHash(uint64_t* vals, uint64_t* hashes, uint64_t count) {
  ankerl::nanobench::Config().minEpochIterations(ITER).run(
          fmt::format("[SIMD OFF] hash-{}-values", count), [&] {
            hash::hash(vals, hashes, count);
          });

  // sample some hashes for correctness check
  static constexpr uint64_t SAMPLE = 1000;
  uint64_t sample_ids[SAMPLE], simd_off_samples[SAMPLE];
  std::random_device rd;
  std::mt19937 gen(rd());
  auto dist = std::uniform_int_distribution<uint64_t>(0, count - 1);
  for (uint64_t i = 0; i < SAMPLE; ++i) {
    sample_ids[i] = dist(gen);
    simd_off_samples[i] = hashes[sample_ids[i]];
  }

  ankerl::nanobench::Config().minEpochIterations(ITER).run(
          fmt::format("[AVX2] hash-{}-values", count), [&] {
            hash::hash_avx2(vals, hashes, count);
          });

  // correctness check
  for (uint64_t i = 0; i < SAMPLE; ++i) {
    assert(simd_off_samples[i] == hashes[sample_ids[i]]);
  }
}

int main() {
  printf("Bench hash:\n");

  // data
  const int num_tests = 5;
  uint64_t counts[num_tests] = {1000, 10000, 100000, 1000000, 10000000};
  uint64_t* vals[num_tests];
  uint64_t* hashes[num_tests];
  for (int i = 0; i < num_tests; ++i) {
    vals[i] = new uint64_t[counts[i]];
    hashes[i] = new uint64_t[counts[i]];
    makeInput(vals[i], counts[i]);
  }

  // bench
  for (int i = 0; i < num_tests; ++i) {
    runHash(vals[i], hashes[i], counts[i]);
  }

  // clear
  for (int i = 0; i < num_tests; ++i) {
    delete[] vals[i];
    delete[] hashes[i];
  }

  return 0;
}
