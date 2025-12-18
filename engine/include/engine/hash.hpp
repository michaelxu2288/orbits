#pragma once
#include <bit>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace eng::core {

constexpr std::uint64_t fnv1a64(std::string_view s) noexcept {
    std::uint64_t h = 14695981039346656037ull;
    for (unsigned char c : s) {
        h ^= static_cast<std::uint64_t>(c);
        h *= 1099511628211ull;
    }
    return h;
}

template<class T>
constexpr std::uint64_t bit_hash(const T& x) noexcept
requires (std::is_trivially_copyable_v<T>)
{
    std::uint64_t h = 14695981039346656037ull;
    auto bytes = std::as_bytes(std::span{&x, 1});
    for (auto b : bytes) {
        h ^= static_cast<std::uint64_t>(b);
        h *= 1099511628211ull;
    }
    return h;
}

constexpr std::uint64_t mix64(std::uint64_t x) noexcept {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return x;
}

}
