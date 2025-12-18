#pragma once
#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <optional>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

namespace eng::algo {

template<class R>
concept range = std::ranges::range<R>;

template<class R>
using range_value_t = std::ranges::range_value_t<R>;

template<range R, class Proj = std::identity>
requires std::regular_invocable<Proj&, range_value_t<R>&>
auto stable_hash(R&& r, Proj proj = {}) {
    using T = std::remove_cvref_t<decltype(std::invoke(proj, *std::ranges::begin(r)))>;
    std::uint64_t h = 14695981039346656037ull;
    for (auto&& e : r) {
        auto x = std::invoke(proj, e);
        if constexpr (std::is_integral_v<T>) {
            h ^= static_cast<std::uint64_t>(x);
            h *= 1099511628211ull;
        } else if constexpr (std::is_floating_point_v<T>) {
            std::uint64_t b = std::bit_cast<std::uint64_t>(static_cast<double>(x));
            h ^= b;
            h *= 1099511628211ull;
        } else {
            h ^= reinterpret_cast<std::uintptr_t>(&x);
            h *= 1099511628211ull;
        }
    }
    return h;
}

template<std::floating_point T>
struct online_stats final {
    std::size_t n{};
    T mean{};
    T m2{};

    void push(T x) {
        ++n;
        auto d = x - mean;
        mean += d / static_cast<T>(n);
        auto d2 = x - mean;
        m2 += d * d2;
    }

    T variance() const {
        return n > 1 ? (m2 / static_cast<T>(n-1)) : T{};
    }

    T stdev() const {
        return std::sqrt(static_cast<long double>(variance()));
    }
};

template<std::integral I>
constexpr I ceil_pow2(I x) {
    if (x <= 1) return 1;
    using U = std::make_unsigned_t<I>;
    U u = static_cast<U>(x-1);
    u |= u >> 1;
    u |= u >> 2;
    u |= u >> 4;
    if constexpr (sizeof(U) >= 2) u |= u >> 8;
    if constexpr (sizeof(U) >= 4) u |= u >> 16;
    if constexpr (sizeof(U) >= 8) u |= u >> 32;
    return static_cast<I>(u + 1);
}

template<class T>
concept sortable = requires(T a, T b) { a < b; };

template<sortable T>
std::vector<T> topk(std::span<const T> s, std::size_t k) {
    std::vector<T> v(s.begin(), s.end());
    if (k >= v.size()) {
        std::sort(v.begin(), v.end(), std::greater<>{});
        return v;
    }
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end(), std::greater<>{});
    v.resize(k);
    std::sort(v.begin(), v.end(), std::greater<>{});
    return v;
}

}
