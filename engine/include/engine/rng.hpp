#pragma once
#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace eng::math {

struct splitmix64 final {
    std::uint64_t x{};

    constexpr explicit splitmix64(std::uint64_t seed = 0x9e3779b97f4a7c15ull) : x(seed) {}

    constexpr std::uint64_t next_u64() noexcept {
        std::uint64_t z = (x += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }

    constexpr std::uint32_t next_u32() noexcept {
        return static_cast<std::uint32_t>(next_u64() >> 32);
    }

    template<class T>
    requires std::is_floating_point_v<T>
    T uniform01() noexcept {
        constexpr int bits = std::numeric_limits<std::uint64_t>::digits;
        auto u = next_u64();
        auto mant = u >> (bits - 53);
        auto d = static_cast<long double>(mant) / static_cast<long double>(1ull << 53);
        return static_cast<T>(d);
    }

    template<class T>
    requires std::is_integral_v<T>
    T uniform(T lo, T hi) noexcept {
        auto span = static_cast<std::uint64_t>(hi - lo);
        if (span == 0) return lo;
        auto r = next_u64();
        auto m = static_cast<std::uint64_t>((__uint128_t(r) * __uint128_t(span + 1)) >> 64);
        return static_cast<T>(lo + static_cast<T>(m));
    }
};

}
