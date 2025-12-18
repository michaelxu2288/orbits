#pragma once
#include <array>
#include <bit>
#include <cmath>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

namespace eng::math {

template<class T>
concept scalar = std::is_arithmetic_v<T> && (!std::is_same_v<T, bool>);

template<scalar T, std::size_t N>
struct vec final {
    std::array<T, N> v{};

    constexpr vec() = default;

    template<class... U>
    requires (sizeof...(U) == N) && (std::convertible_to<U, T> && ...)
    constexpr explicit vec(U&&... u) : v{static_cast<T>(std::forward<U>(u))...} {}

    constexpr T& operator[](std::size_t i) noexcept { return v[i]; }
    constexpr const T& operator[](std::size_t i) const noexcept { return v[i]; }

    constexpr auto operator<=>(const vec&) const = default;

    constexpr vec operator+() const noexcept { return *this; }
    constexpr vec operator-() const noexcept {
        vec r;
        for (std::size_t i = 0; i < N; ++i) r.v[i] = -v[i];
        return r;
    }

    constexpr vec& operator+=(const vec& o) noexcept {
        for (std::size_t i = 0; i < N; ++i) v[i] += o.v[i];
        return *this;
    }
    constexpr vec& operator-=(const vec& o) noexcept {
        for (std::size_t i = 0; i < N; ++i) v[i] -= o.v[i];
        return *this;
    }
    constexpr vec& operator*=(T s) noexcept {
        for (std::size_t i = 0; i < N; ++i) v[i] *= s;
        return *this;
    }
    constexpr vec& operator/=(T s) noexcept {
        for (std::size_t i = 0; i < N; ++i) v[i] /= s;
        return *this;
    }

    friend constexpr vec operator+(vec a, const vec& b) noexcept { return a += b; }
    friend constexpr vec operator-(vec a, const vec& b) noexcept { return a -= b; }
    friend constexpr vec operator*(vec a, T s) noexcept { return a *= s; }
    friend constexpr vec operator*(T s, vec a) noexcept { return a *= s; }
    friend constexpr vec operator/(vec a, T s) noexcept { return a /= s; }

    constexpr T dot(const vec& o) const noexcept {
        T r{};
        for (std::size_t i = 0; i < N; ++i) r += v[i] * o.v[i];
        return r;
    }

    constexpr T norm2() const noexcept { return dot(*this); }

    T norm() const noexcept { return std::sqrt(static_cast<long double>(norm2())); }

    vec normalized(T eps = std::numeric_limits<T>::epsilon()) const noexcept {
        auto n = norm();
        if (n <= eps) return *this;
        return (*this) / static_cast<T>(n);
    }

    constexpr std::span<T, N> span() noexcept { return std::span<T, N>(v); }
    constexpr std::span<const T, N> span() const noexcept { return std::span<const T, N>(v); }
};

template<scalar T> using vec2 = vec<T, 2>;
template<scalar T> using vec3 = vec<T, 3>;
template<scalar T> using vec4 = vec<T, 4>;

template<scalar T>
constexpr vec3<T> cross(const vec3<T>& a, const vec3<T>& b) noexcept {
    return vec3<T>{
        a.v[1]*b.v[2] - a.v[2]*b.v[1],
        a.v[2]*b.v[0] - a.v[0]*b.v[2],
        a.v[0]*b.v[1] - a.v[1]*b.v[0]
    };
}

template<scalar T, std::size_t N>
constexpr vec<T,N> hadamard(const vec<T,N>& a, const vec<T,N>& b) noexcept {
    vec<T,N> r;
    for (std::size_t i = 0; i < N; ++i) r.v[i] = a.v[i]*b.v[i];
    return r;
}

template<scalar T, std::size_t N>
constexpr vec<T,N> clamp(const vec<T,N>& x, T lo, T hi) noexcept {
    vec<T,N> r;
    for (std::size_t i = 0; i < N; ++i) r.v[i] = std::clamp(x.v[i], lo, hi);
    return r;
}

}
