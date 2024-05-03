//
// Created by Yifei Yang on 5/2/24.
//

#ifndef ARROW_BF_SIMD_BENCH_HASH_HASH_HPP
#define ARROW_BF_SIMD_BENCH_HASH_HASH_HPP

#include <cstdint>
#include <immintrin.h>

namespace hash {

static const uint64_t CONST = 0xd6e8feb86659fd93U;

inline uint64_t hash(uint64_t x) {
  x ^= x >> 32;
  x *= CONST;
  x ^= x >> 32;
  x *= CONST;
  x ^= x >> 32;
  return x;
}

inline void hash(uint64_t* vals, uint64_t* hashes, uint64_t count) {
  for (uint64_t i = 0; i < count; ++i) {
    hashes[i] = hash(vals[i]);
  }
}

inline __m256i hash_avx2(__m256i x) {
  x = _mm256_xor_si256(x, _mm256_srli_epi64(x, 32));
  x = _mm256_mullo_epi64(x, _mm256_set1_epi64x(CONST));
  x = _mm256_xor_si256(x, _mm256_srli_epi64(x, 32));
  x = _mm256_mullo_epi64(x, _mm256_set1_epi64x(CONST));
  x = _mm256_xor_si256(x, _mm256_srli_epi64(x, 32));
  return x;
}

inline void hash_avx2(uint64_t* vals, uint64_t* hashes, uint64_t count) {
  constexpr int unroll = 4;
  for (uint64_t i = 0; i < count / unroll; ++i) {
    __m256i res = hash_avx2(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(vals) + i));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(hashes) + i, res);
  }
  for (uint64_t i = count - (count % unroll); i < count; ++i) {
    hashes[i] = hash(vals[i]);
  }
}

}

#endif //ARROW_BF_SIMD_BENCH_HASH_HASH_HPP
