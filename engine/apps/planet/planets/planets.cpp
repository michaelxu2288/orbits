

// //#include <algorithm>
// #include <array>
// #include <bit>
// #include <chrono>
// #include <cmath>
// #include <concepts>
// #include <cstddef>
// #include <cstdint>
// #include <expected>
// #include <filesystem>
// #include <fstream>
// #include <functional>
// #include <iomanip>
// #include <iostream>
// #include <limits>
// #include <numbers>
// #include <optional>
// #include <ranges>
// #include <span>
// #include <string>
// #include <string_view>
// #include <tuple>
// #include <type_traits>
// #include <utility>
// #include <variant>
// #include <vector>

// namespace planet {

// using u8  = std::uint8_t;
// using u16 = std::uint16_t;
// using u32 = std::uint32_t;
// using u64 = std::uint64_t;
// using i32 = std::int32_t;
// using i64 = std::int64_t;
// using f32 = float;
// using f64 = double;

// static constexpr std::string_view kFileTag = "planet.cpcp";

// static constexpr i32 kZeroI = i32{0};
// static constexpr i32 kOneI  = i32{1};
// static constexpr i32 kTwoI  = i32{2};
// static constexpr i32 kThreeI = i32{3};
// static constexpr i32 kFourI = i32{4};
// static constexpr i32 kFiveI = i32{5};
// static constexpr i32 kSixI = i32{6};
// static constexpr i32 kSevenI = i32{7};
// static constexpr i32 kEightI = i32{8};
// static constexpr i32 kNineI = i32{9};

// static constexpr u32 kZeroU = u32{0};
// static constexpr u32 kOneU  = u32{1};
// static constexpr u32 kTwoU  = u32{2};
// static constexpr u32 kThreeU = u32{3};
// static constexpr u32 kFourU = u32{4};

// static constexpr f64 kZero = f64{0.0};
// static constexpr f64 kOne  = f64{1.0};
// static constexpr f64 kTwo  = f64{2.0};
// static constexpr f64 kThree = f64{3.0};
// static constexpr f64 kFour = f64{4.0};
// static constexpr f64 kHalf = f64{0.5};
// static constexpr f64 kQuarter = f64{0.25};
// static constexpr f64 kEighth = f64{0.125};

// static constexpr f64 kPi  = std::numbers::pi_v<f64>;
// static constexpr f64 kTau = kTwo * kPi;
// static constexpr f64 kDegToRad = kPi / f64{180.0};
// static constexpr f64 kRadToDeg = f64{180.0} / kPi;

// static constexpr f64 kEpsilon = std::numeric_limits<f64>::epsilon();
// static constexpr f64 kTiny = f64{1e-12};
// static constexpr f64 kSafeMin = f64{1e-30};
// static constexpr f64 kSafeMax = f64{1e+30};

// template<class T>
// concept Scalar = std::is_arithmetic_v<T> && (!std::is_same_v<T,bool>);

// template<Scalar T, std::size_t N>
// struct vec final {
//     std::array<T, N> v{};

//     constexpr vec() = default;

//     template<class... U>
//     requires (sizeof...(U) == N) && (std::convertible_to<U, T> && ...)
//     constexpr explicit vec(U&&... u) : v{static_cast<T>(std::forward<U>(u))...} {}

//     constexpr T& operator[](std::size_t i) noexcept { return v[i]; }
//     constexpr const T& operator[](std::size_t i) const noexcept { return v[i]; }

//     constexpr auto operator<=>(const vec&) const = default;

//     constexpr vec operator+() const noexcept { return *this; }
//     constexpr vec operator-() const noexcept {
//         vec r;
//         for (std::size_t i = 0; i < N; ++i) r.v[i] = -v[i];
//         return r;
//     }

//     constexpr vec& operator+=(const vec& o) noexcept {
//         for (std::size_t i = 0; i < N; ++i) v[i] += o.v[i];
//         return *this;
//     }

//     constexpr vec& operator-=(const vec& o) noexcept {
//         for (std::size_t i = 0; i < N; ++i) v[i] -= o.v[i];
//         return *this;
//     }

//     constexpr vec& operator*=(T s) noexcept {
//         for (std::size_t i = 0; i < N; ++i) v[i] *= s;
//         return *this;
//     }

//     constexpr vec& operator/=(T s) noexcept {
//         for (std::size_t i = 0; i < N; ++i) v[i] /= s;
//         return *this;
//     }

//     friend constexpr vec operator+(vec a, const vec& b) noexcept { return a += b; }
//     friend constexpr vec operator-(vec a, const vec& b) noexcept { return a -= b; }
//     friend constexpr vec operator*(vec a, T s) noexcept { return a *= s; }
//     friend constexpr vec operator*(T s, vec a) noexcept { return a *= s; }
//     friend constexpr vec operator/(vec a, T s) noexcept { return a /= s; }

//     constexpr T dot(const vec& o) const noexcept {
//         T r{};
//         for (std::size_t i = 0; i < N; ++i) r += v[i] * o.v[i];
//         return r;
//     }

//     constexpr T norm2() const noexcept { return dot(*this); }

//     T norm() const noexcept {
//         return std::sqrt(static_cast<long double>(norm2()));
//     }

//     vec normalized(T eps = static_cast<T>(kEpsilon)) const noexcept {
//         auto n = norm();
//         if (n <= eps) return *this;
//         return (*this) / static_cast<T>(n);
//     }
// };

// template<Scalar T> using vec2 = vec<T,2>;
// template<Scalar T> using vec3 = vec<T,3>;
// template<Scalar T> using vec4 = vec<T,4>;

// static constexpr vec3<f64> kAxisX{ kOne, kZero, kZero };
// static constexpr vec3<f64> kAxisY{ kZero, kOne, kZero };
// static constexpr vec3<f64> kAxisZ{ kZero, kZero, kOne };

// template<Scalar T>
// constexpr vec3<T> cross(const vec3<T>& a, const vec3<T>& b) noexcept {
//     return vec3<T>{
//         a.v[1]*b.v[2] - a.v[2]*b.v[1],
//         a.v[2]*b.v[0] - a.v[0]*b.v[2],
//         a.v[0]*b.v[1] - a.v[1]*b.v[0]
//     };
// }

// template<Scalar T>
// constexpr T clamp(T x, T lo, T hi) noexcept {
//     return (x < lo) ? lo : ((x > hi) ? hi : x);
// }

// template<Scalar T>
// constexpr T lerp(T a, T b, T t) noexcept {
//     return a + (b - a) * t;
// }

// constexpr u64 mix64(u64 x) noexcept {
//     x ^= (x >> u64{33});
//     x *= u64{0xff51afd7ed558ccdULL};
//     x ^= (x >> u64{33});
//     x *= u64{0xc4ceb9fe1a85ec53ULL};
//     x ^= (x >> u64{33});
//     return x;
// }

// struct splitmix64 final {
//     u64 x{};
//     constexpr explicit splitmix64(u64 seed) : x(seed) {}

//     constexpr u64 next_u64() noexcept {
//         x += u64{0x9e3779b97f4a7c15ULL};
//         u64 z = x;
//         z = (z ^ (z >> u64{30})) * u64{0xbf58476d1ce4e5b9ULL};
//         z = (z ^ (z >> u64{27})) * u64{0x94d049bb133111ebULL};
//         return z ^ (z >> u64{31});
//     }

//     f64 uniform01() noexcept {
//         constexpr u64 kMantBits = u64{53};
//         constexpr u64 kShift = u64{64} - kMantBits;
//         u64 u = next_u64() >> kShift;
//         f64 denom = static_cast<f64>(u64{1} << kMantBits);
//         return static_cast<f64>(u) / denom;
//     }

//     f64 uniform(f64 lo, f64 hi) noexcept {
//         return lerp(lo, hi, uniform01());
//     }
// };

// constexpr u32 pack_rgba8(vec4<f64> c) noexcept {
//     auto to8 = [](f64 x) constexpr -> u32 {
//         f64 y = clamp(x, kZero, kOne);
//         f64 scaled = y * static_cast<f64>(u32{255});
//         return static_cast<u32>(scaled + kHalf);
//     };
//     u32 r = to8(c[0]);
//     u32 g = to8(c[1]);
//     u32 b = to8(c[2]);
//     u32 a = to8(c[3]);
//     return (a << u32{24}) | (b << u32{16}) | (g << u32{8}) | r;
// }

// constexpr vec4<f64> unpack_rgba8(u32 p) noexcept {
//     auto f = [](u32 x) constexpr -> f64 { return static_cast<f64>(x) / static_cast<f64>(u32{255}); };
//     u32 r = (p >> u32{0})  & u32{255};
//     u32 g = (p >> u32{8})  & u32{255};
//     u32 b = (p >> u32{16}) & u32{255};
//     u32 a = (p >> u32{24}) & u32{255};
//     return vec4<f64>{ f(r), f(g), f(b), f(a) };
// }

// struct mat3 final {
//     vec3<f64> c0{ kOne, kZero, kZero };
//     vec3<f64> c1{ kZero, kOne, kZero };
//     vec3<f64> c2{ kZero, kZero, kOne };

//     static mat3 from_cols(vec3<f64> a, vec3<f64> b, vec3<f64> c) noexcept {
//         mat3 m;
//         m.c0 = a; m.c1 = b; m.c2 = c;
//         return m;
//     }

//     vec3<f64> mul(vec3<f64> v) const noexcept {
//         return c0 * v[0] + c1 * v[1] + c2 * v[2];
//     }
// };

// mat3 basis_from_forward(vec3<f64> fwd, vec3<f64> up_hint) noexcept {
//     auto f = fwd.normalized();
//     auto r = cross(up_hint, f);
//     if (r.norm2() <= kTiny) r = cross(kAxisY, f);
//     r = r.normalized();
//     auto u = cross(f, r).normalized();
//     return mat3::from_cols(r, u, f);
// }

// struct camera final {
//     vec3<f64> pos{ kZero, kZero, kZero };
//     vec3<f64> look{ kZero, kZero, kOne };
//     vec3<f64> up{ kZero, kOne, kZero };
//     f64 vfov_rad{ f64{60.0} * kDegToRad };
//     f64 aspect{ f64{16.0} / f64{9.0} };

//     mat3 view_basis() const noexcept {
//         return basis_from_forward(look - pos, up);
//     }
// };

// struct surface_palette final {
//     vec4<f64> deep_ocean{ f64{0.02}, f64{0.08}, f64{0.18}, kOne };
//     vec4<f64> shallow_ocean{ f64{0.02}, f64{0.18}, f64{0.24}, kOne };
//     vec4<f64> beach{ f64{0.76}, f64{0.70}, f64{0.52}, kOne };
//     vec4<f64> lowland{ f64{0.10}, f64{0.35}, f64{0.18}, kOne };
//     vec4<f64> highland{ f64{0.22}, f64{0.40}, f64{0.26}, kOne };
//     vec4<f64> rock{ f64{0.42}, f64{0.42}, f64{0.44}, kOne };
//     vec4<f64> snow{ f64{0.92}, f64{0.94}, f64{0.98}, kOne };
//     vec4<f64> atmosphere{ f64{0.30}, f64{0.60}, f64{0.95}, f64{0.65} };
// };

// struct planet_descriptor final {
//     u64 seed{ u64{0xC0FFEEULL} };
//     f64 radius{ f64{6.371e6} };
//     f64 ocean_level{ f64{0.35} };
//     f64 mountain_amp{ f64{0.18} };
//     f64 continental_amp{ f64{0.55} };
//     f64 roughness{ f64{0.85} };
//     f64 atmosphere_height{ f64{0.06} };
//     f64 axial_tilt_rad{ f64{23.44} * kDegToRad };
//     f64 sun_intensity{ f64{1.25} };
//     vec3<f64> sun_dir{ f64{0.4}, f64{0.3}, f64{0.86} };
//     surface_palette palette{};
// };

// struct orbit_params final {
//     f64 semi_major{ f64{1.0} };
//     f64 eccentricity{ f64{0.0} };
//     f64 inclination_rad{ kZero };
//     f64 asc_node_rad{ kZero };
//     f64 arg_periapsis_rad{ kZero };
//     f64 mean_anomaly0_rad{ kZero };
//     f64 mu_central{ f64{1.0} };
// };

// struct vertex final {
//     vec3<f64> pos{};
//     vec3<f64> nrm{};
//     vec2<f64> uv{};
//     vec4<f64> albedo{};
//     f64 height{};
//     f64 humidity{};
//     f64 temperature{};
// };

// struct mesh final {
//     std::vector<vertex> vtx;
//     std::vector<u32> idx;
// };

// constexpr f64 smoothstep(f64 e0, f64 e1, f64 x) noexcept {
//     f64 t = clamp((x - e0) / (e1 - e0 + kTiny), kZero, kOne);
//     return t * t * (kThree - kTwo * t);
// }

// constexpr f64 remap01(f64 x, f64 lo, f64 hi) noexcept {
//     return clamp((x - lo) / (hi - lo + kTiny), kZero, kOne);
// }

// constexpr f64 s_curve(f64 t) noexcept {
//     return t * t * (kThree - kTwo * t);
// }

// static inline u64 hash_grid(i64 x, i64 y, i64 z, u64 seed) noexcept {
//     u64 h = seed;
//     h ^= mix64(static_cast<u64>(x) + u64{0x9e3779b97f4a7c15ULL});
//     h ^= mix64(static_cast<u64>(y) + u64{0xbf58476d1ce4e5b9ULL});
//     h ^= mix64(static_cast<u64>(z) + u64{0x94d049bb133111ebULL});
//     return mix64(h);
// }

// static inline f64 grad(u64 h, f64 x, f64 y, f64 z) noexcept {
//     constexpr u64 kMask = u64{15};
//     u64 g = h & kMask;
//     f64 u = (g < u64{8}) ? x : y;
//     f64 v = (g < u64{4}) ? y : ((g == u64{12} || g == u64{14}) ? x : z);
//     f64 s0 = ((g & u64{1}) == u64{0}) ? u : -u;
//     f64 s1 = ((g & u64{2}) == u64{0}) ? v : -v;
//     return s0 + s1;
// }

// static inline f64 fade(f64 t) noexcept {
//     return t * t * t * (t * (t * f64{6.0} - f64{15.0}) + f64{10.0});
// }

// static inline f64 noise3(vec3<f64> p, u64 seed) noexcept {
//     f64 fx = std::floor(p[0]);
//     f64 fy = std::floor(p[1]);
//     f64 fz = std::floor(p[2]);

//     i64 X = static_cast<i64>(fx);
//     i64 Y = static_cast<i64>(fy);
//     i64 Z = static_cast<i64>(fz);

//     f64 x = p[0] - fx;
//     f64 y = p[1] - fy;
//     f64 z = p[2] - fz;

//     f64 u = fade(x);
//     f64 v = fade(y);
//     f64 w = fade(z);

//     auto H = [&](i64 xi, i64 yi, i64 zi) -> u64 { return hash_grid(xi, yi, zi, seed); };

//     f64 n000 = grad(H(X,   Y,   Z  ), x,     y,     z    );
//     f64 n100 = grad(H(X+1, Y,   Z  ), x-kOne, y,     z    );
//     f64 n010 = grad(H(X,   Y+1, Z  ), x,     y-kOne, z    );
//     f64 n110 = grad(H(X+1, Y+1, Z  ), x-kOne, y-kOne, z   );
//     f64 n001 = grad(H(X,   Y,   Z+1), x,     y,     z-kOne);
//     f64 n101 = grad(H(X+1, Y,   Z+1), x-kOne, y,     z-kOne);
//     f64 n011 = grad(H(X,   Y+1, Z+1), x,     y-kOne, z-kOne);
//     f64 n111 = grad(H(X+1, Y

```cpp
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace planet {

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using f32 = float;
using f64 = double;

static constexpr std::string_view kFileTag = "planet.cpcp";

static constexpr i32 kZeroI = i32{0};
static constexpr i32 kOneI  = i32{1};
static constexpr i32 kTwoI  = i32{2};
static constexpr i32 kThreeI = i32{3};
static constexpr i32 kFourI = i32{4};
static constexpr i32 kFiveI = i32{5};
static constexpr i32 kSixI = i32{6};
static constexpr i32 kSevenI = i32{7};
static constexpr i32 kEightI = i32{8};
static constexpr i32 kNineI = i32{9};

static constexpr u32 kZeroU = u32{0};
static constexpr u32 kOneU  = u32{1};
static constexpr u32 kTwoU  = u32{2};
static constexpr u32 kThreeU = u32{3};
static constexpr u32 kFourU = u32{4};

static constexpr f64 kZero = f64{0.0};
static constexpr f64 kOne  = f64{1.0};
static constexpr f64 kTwo  = f64{2.0};
static constexpr f64 kThree = f64{3.0};
static constexpr f64 kFour = f64{4.0};
static constexpr f64 kHalf = f64{0.5};
static constexpr f64 kQuarter = f64{0.25};
static constexpr f64 kEighth = f64{0.125};

static constexpr f64 kPi  = std::numbers::pi_v<f64>;
static constexpr f64 kTau = kTwo * kPi;
static constexpr f64 kDegToRad = kPi / f64{180.0};
static constexpr f64 kRadToDeg = f64{180.0} / kPi;

static constexpr f64 kEpsilon = std::numeric_limits<f64>::epsilon();
static constexpr f64 kTiny = f64{1e-12};
static constexpr f64 kSafeMin = f64{1e-30};
static constexpr f64 kSafeMax = f64{1e+30};

template<class T>
concept Scalar = std::is_arithmetic_v<T> && (!std::is_same_v<T,bool>);

template<Scalar T, std::size_t N>
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

    T norm() const noexcept {
        return std::sqrt(static_cast<long double>(norm2()));
    }

    vec normalized(T eps = static_cast<T>(kEpsilon)) const noexcept {
        auto n = norm();
        if (n <= eps) return *this;
        return (*this) / static_cast<T>(n);
    }
};

template<Scalar T> using vec2 = vec<T,2>;
template<Scalar T> using vec3 = vec<T,3>;
template<Scalar T> using vec4 = vec<T,4>;

static constexpr vec3<f64> kAxisX{ kOne, kZero, kZero };
static constexpr vec3<f64> kAxisY{ kZero, kOne, kZero };
static constexpr vec3<f64> kAxisZ{ kZero, kZero, kOne };

template<Scalar T>
constexpr vec3<T> cross(const vec3<T>& a, const vec3<T>& b) noexcept {
    return vec3<T>{
        a.v[1]*b.v[2] - a.v[2]*b.v[1],
        a.v[2]*b.v[0] - a.v[0]*b.v[2],
        a.v[0]*b.v[1] - a.v[1]*b.v[0]
    };
}

template<Scalar T>
constexpr T clamp(T x, T lo, T hi) noexcept {
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}

template<Scalar T>
constexpr T lerp(T a, T b, T t) noexcept {
    return a + (b - a) * t;
}

constexpr u64 mix64(u64 x) noexcept {
    x ^= (x >> u64{33});
    x *= u64{0xff51afd7ed558ccdULL};
    x ^= (x >> u64{33});
    x *= u64{0xc4ceb9fe1a85ec53ULL};
    x ^= (x >> u64{33});
    return x;
}

struct splitmix64 final {
    u64 x{};
    constexpr explicit splitmix64(u64 seed) : x(seed) {}

    constexpr u64 next_u64() noexcept {
        x += u64{0x9e3779b97f4a7c15ULL};
        u64 z = x;
        z = (z ^ (z >> u64{30})) * u64{0xbf58476d1ce4e5b9ULL};
        z = (z ^ (z >> u64{27})) * u64{0x94d049bb133111ebULL};
        return z ^ (z >> u64{31});
    }

    f64 uniform01() noexcept {
        constexpr u64 kMantBits = u64{53};
        constexpr u64 kShift = u64{64} - kMantBits;
        u64 u = next_u64() >> kShift;
        f64 denom = static_cast<f64>(u64{1} << kMantBits);
        return static_cast<f64>(u) / denom;
    }

    f64 uniform(f64 lo, f64 hi) noexcept {
        return lerp(lo, hi, uniform01());
    }
};

constexpr u32 pack_rgba8(vec4<f64> c) noexcept {
    auto to8 = [](f64 x) constexpr -> u32 {
        f64 y = clamp(x, kZero, kOne);
        f64 scaled = y * static_cast<f64>(u32{255});
        return static_cast<u32>(scaled + kHalf);
    };
    u32 r = to8(c[0]);
    u32 g = to8(c[1]);
    u32 b = to8(c[2]);
    u32 a = to8(c[3]);
    return (a << u32{24}) | (b << u32{16}) | (g << u32{8}) | r;
}

constexpr vec4<f64> unpack_rgba8(u32 p) noexcept {
    auto f = [](u32 x) constexpr -> f64 { return static_cast<f64>(x) / static_cast<f64>(u32{255}); };
    u32 r = (p >> u32{0})  & u32{255};
    u32 g = (p >> u32{8})  & u32{255};
    u32 b = (p >> u32{16}) & u32{255};
    u32 a = (p >> u32{24}) & u32{255};
    return vec4<f64>{ f(r), f(g), f(b), f(a) };
}

struct mat3 final {
    vec3<f64> c0{ kOne, kZero, kZero };
    vec3<f64> c1{ kZero, kOne, kZero };
    vec3<f64> c2{ kZero, kZero, kOne };

    static mat3 from_cols(vec3<f64> a, vec3<f64> b, vec3<f64> c) noexcept {
        mat3 m;
        m.c0 = a; m.c1 = b; m.c2 = c;
        return m;
    }

    vec3<f64> mul(vec3<f64> v) const noexcept {
        return c0 * v[0] + c1 * v[1] + c2 * v[2];
    }
};

mat3 basis_from_forward(vec3<f64> fwd, vec3<f64> up_hint) noexcept {
    auto f = fwd.normalized();
    auto r = cross(up_hint, f);
    if (r.norm2() <= kTiny) r = cross(kAxisY, f);
    r = r.normalized();
    auto u = cross(f, r).normalized();
    return mat3::from_cols(r, u, f);
}

struct camera final {
    vec3<f64> pos{ kZero, kZero, kZero };
    vec3<f64> look{ kZero, kZero, kOne };
    vec3<f64> up{ kZero, kOne, kZero };
    f64 vfov_rad{ f64{60.0} * kDegToRad };
    f64 aspect{ f64{16.0} / f64{9.0} };

    mat3 view_basis() const noexcept {
        return basis_from_forward(look - pos, up);
    }
};

struct surface_palette final {
    vec4<f64> deep_ocean{ f64{0.02}, f64{0.08}, f64{0.18}, kOne };
    vec4<f64> shallow_ocean{ f64{0.02}, f64{0.18}, f64{0.24}, kOne };
    vec4<f64> beach{ f64{0.76}, f64{0.70}, f64{0.52}, kOne };
    vec4<f64> lowland{ f64{0.10}, f64{0.35}, f64{0.18}, kOne };
    vec4<f64> highland{ f64{0.22}, f64{0.40}, f64{0.26}, kOne };
    vec4<f64> rock{ f64{0.42}, f64{0.42}, f64{0.44}, kOne };
    vec4<f64> snow{ f64{0.92}, f64{0.94}, f64{0.98}, kOne };
    vec4<f64> atmosphere{ f64{0.30}, f64{0.60}, f64{0.95}, f64{0.65} };
};

struct planet_descriptor final {
    u64 seed{ u64{0xC0FFEEULL} };
    f64 radius{ f64{6.371e6} };
    f64 ocean_level{ f64{0.35} };
    f64 mountain_amp{ f64{0.18} };
    f64 continental_amp{ f64{0.55} };
    f64 roughness{ f64{0.85} };
    f64 atmosphere_height{ f64{0.06} };
    f64 axial_tilt_rad{ f64{23.44} * kDegToRad };
    f64 sun_intensity{ f64{1.25} };
    vec3<f64> sun_dir{ f64{0.4}, f64{0.3}, f64{0.86} };
    surface_palette palette{};
};

struct orbit_params final {
    f64 semi_major{ f64{1.0} };
    f64 eccentricity{ f64{0.0} };
    f64 inclination_rad{ kZero };
    f64 asc_node_rad{ kZero };
    f64 arg_periapsis_rad{ kZero };
    f64 mean_anomaly0_rad{ kZero };
    f64 mu_central{ f64{1.0} };
};

struct vertex final {
    vec3<f64> pos{};
    vec3<f64> nrm{};
    vec2<f64> uv{};
    vec4<f64> albedo{};
    f64 height{};
    f64 humidity{};
    f64 temperature{};
};

struct mesh final {
    std::vector<vertex> vtx;
    std::vector<u32> idx;
};

constexpr f64 smoothstep(f64 e0, f64 e1, f64 x) noexcept {
    f64 t = clamp((x - e0) / (e1 - e0 + kTiny), kZero, kOne);
    return t * t * (kThree - kTwo * t);
}

constexpr f64 remap01(f64 x, f64 lo, f64 hi) noexcept {
    return clamp((x - lo) / (hi - lo + kTiny), kZero, kOne);
}

constexpr f64 s_curve(f64 t) noexcept {
    return t * t * (kThree - kTwo * t);
}

static inline u64 hash_grid(i64 x, i64 y, i64 z, u64 seed) noexcept {
    u64 h = seed;
    h ^= mix64(static_cast<u64>(x) + u64{0x9e3779b97f4a7c15ULL});
    h ^= mix64(static_cast<u64>(y) + u64{0xbf58476d1ce4e5b9ULL});
    h ^= mix64(static_cast<u64>(z) + u64{0x94d049bb133111ebULL});
    return mix64(h);
}

static inline f64 grad(u64 h, f64 x, f64 y, f64 z) noexcept {
    constexpr u64 kMask = u64{15};
    u64 g = h & kMask;
    f64 u = (g < u64{8}) ? x : y;
    f64 v = (g < u64{4}) ? y : ((g == u64{12} || g == u64{14}) ? x : z);
    f64 s0 = ((g & u64{1}) == u64{0}) ? u : -u;
    f64 s1 = ((g & u64{2}) == u64{0}) ? v : -v;
    return s0 + s1;
}

static inline f64 fade(f64 t) noexcept {
    return t * t * t * (t * (t * f64{6.0} - f64{15.0}) + f64{10.0});
}

static inline f64 noise3(vec3<f64> p, u64 seed) noexcept {
    f64 fx = std::floor(p[0]);
    f64 fy = std::floor(p[1]);
    f64 fz = std::floor(p[2]);

    i64 X = static_cast<i64>(fx);
    i64 Y = static_cast<i64>(fy);
    i64 Z = static_cast<i64>(fz);

    f64 x = p[0] - fx;
    f64 y = p[1] - fy;
    f64 z = p[2] - fz;

    f64 u = fade(x);
    f64 v = fade(y);
    f64 w = fade(z);

    auto H = [&](i64 xi, i64 yi, i64 zi) -> u64 { return hash_grid(xi, yi, zi, seed); };

    f64 n000 = grad(H(X,   Y,   Z  ), x,     y,     z    );
    f64 n100 = grad(H(X+1, Y,   Z  ), x-kOne, y,     z    );
    f64 n010 = grad(H(X,   Y+1, Z  ), x,     y-kOne, z    );
    f64 n110 = grad(H(X+1, Y+1, Z  ), x-kOne, y-kOne, z   );
    f64 n001 = grad(H(X,   Y,   Z+1), x,     y,     z-kOne);
    f64 n101 = grad(H(X+1, Y,   Z+1), x-kOne, y,     z-kOne);
    f64 n011 = grad(H(X,   Y+1, Z+1), x,     y-kOne, z-kOne);
    f64 n111 = grad(H(X+1, Y+1, Z+1), x-kOne, y-kOne, z-kOne);

    f64 nx00 = lerp(n000, n100, u);
    f64 nx10 = lerp(n010, n110, u);
    f64 nx01 = lerp(n001, n101, u);
    f64 nx11 = lerp(n011, n111, u);

    f64 nxy0 = lerp(nx00, nx10, v);
    f64 nxy1 = lerp(nx01, nx11, v);

    f64 nxyz = lerp(nxy0, nxy1, w);
    return nxyz;
}

struct fbm_params final {
    u32 octaves{ u32{7} };
    f64 lacunarity{ f64{2.0} };
    f64 gain{ f64{0.5} };
    f64 base_freq{ f64{1.0} };
    f64 base_amp{ f64{1.0} };
};

static inline f64 fbm(vec3<f64> p, u64 seed, fbm_params fp) noexcept {
    f64 sum = kZero;
    f64 amp = fp.base_amp;
    f64 freq = fp.base_freq;
    for (u32 i = kZeroU; i < fp.octaves; i += kOneU) {
        f64 n = noise3(p * freq, seed ^ mix64(static_cast<u64>(i)));
        sum += n * amp;
        freq *= fp.lacunarity;
        amp *= fp.gain;
    }
    return sum;
}

static inline vec3<f64> spherify(vec3<f64> p) noexcept {
    return p.normalized();
}

static inline vec2<f64> sphere_uv(vec3<f64> n) noexcept {
    f64 u = (std::atan2(n[1], n[0]) + kPi) / kTau;
    f64 v = (std::asin(clamp(n[2], -kOne, kOne)) + kPi * kHalf) / kPi;
    return vec2<f64>{u, v};
}

static inline vec3<f64> rotate_axis_angle(vec3<f64> v, vec3<f64> axis, f64 angle) noexcept {
    auto a = axis.normalized();
    f64 c = std::cos(angle);
    f64 s = std::sin(angle);
    return v * c + cross(a, v) * s + a * (a.dot(v) * (kOne - c));
}

static inline f64 climate_temperature(vec3<f64> n, f64 axial_tilt_rad, vec3<f64> sun_dir) noexcept {
    auto tilt_axis = rotate_axis_angle(kAxisY, kAxisX, axial_tilt_rad).normalized();
    auto equator = cross(tilt_axis, kAxisZ).normalized();
    f64 lat = std::asin(clamp(n.dot(tilt_axis), -kOne, kOne));
    f64 latitude01 = kOne - std::abs(lat) / (kPi * kHalf);
    f64 insolation = clamp(n.dot(sun_dir.normalized()), kZero, kOne);
    f64 t = s_curve(latitude01) * lerp(f64{0.35}, f64{1.0}, insolation);
    return clamp(t, kZero, kOne);
}

static inline f64 climate_humidity(vec3<f64> n, u64 seed) noexcept {
    fbm_params fp{};
    fp.octaves = u32{6};
    fp.lacunarity = f64{2.15};
    fp.gain = f64{0.55};
    fp.base_freq = f64{1.75};
    fp.base_amp = f64{1.0};
    f64 h = fbm(n * f64{3.0}, seed ^ u64{0xA11CEULL}, fp);
    f64 r = remap01(h, -f64{0.9}, f64{0.9});
    return clamp(r, kZero, kOne);
}

static inline f64 terrain_height(vec3<f64> n, const planet_descriptor& d) noexcept {
    fbm_params continents{};
    continents.octaves = u32{5};
    continents.lacunarity = f64{2.0};
    continents.gain = f64{0.5};
    continents.base_freq = f64{0.85};
    continents.base_amp = f64{1.0};

    fbm_params mountains{};
    mountains.octaves = u32{8};
    mountains.lacunarity = f64{2.25};
    mountains.gain = f64{0.48};
    mountains.base_freq = f64{3.25};
    mountains.base_amp = f64{1.0};

    f64 c = fbm(n, d.seed ^ u64{0xC0NT1N3NTULL}, continents);
    f64 m = fbm(n, d.seed ^ u64{0xM0UNT41NULL}, mountains);

    f64 c01 = remap01(c, -f64{0.8}, f64{0.8});
    f64 ridges = kOne - std::abs(m);
    f64 ridgy = std::pow(clamp(ridges, kZero, kOne), lerp(f64{1.25}, f64{3.75}, d.roughness));

    f64 base = (c01 - d.ocean_level) * d.continental_amp;
    f64 peaks = (ridgy - f64{0.45}) * d.mountain_amp;
    f64 h = base + peaks;

    f64 cap = f64{0.95};
    h = clamp(h, -cap, cap);
    return h;
}

static inline vec4<f64> shade_surface(const planet_descriptor& d, f64 h, f64 temp, f64 humid) noexcept {
    auto& P = d.palette;
    f64 ocean = smoothstep(-f64{0.65}, -f64{0.05}, h);
    f64 shore = smoothstep(-f64{0.05}, f64{0.02}, h);
    f64 low = smoothstep(f64{0.02}, f64{0.18}, h);
    f64 high = smoothstep(f64{0.18}, f64{0.42}, h);
    f64 peak = smoothstep(f64{0.42}, f64{0.70}, h);

    vec4<f64> c = P.deep_ocean;
    c = vec4<f64>{
        lerp(c[0], P.shallow_ocean[0], ocean),
        lerp(c[1], P.shallow_ocean[1], ocean),
        lerp(c[2], P.shallow_ocean[2], ocean),
        kOne
    };

    vec4<f64> land = P.lowland;
    vec4<f64> lush = vec4<f64>{ P.lowland[0]*f64{0.85}, P.lowland[1]*f64{1.10}, P.lowland[2]*f64{0.80}, kOne };
    vec4<f64> arid = vec4<f64>{ P.lowland[0]*f64{1.10}, P.lowland[1]*f64{0.95}, P.lowland[2]*f64{0.70}, kOne };
    f64 wet = smoothstep(f64{0.35}, f64{0.85}, humid);
    land = vec4<f64>{
        lerp(arid[0], lush[0], wet),
        lerp(arid[1], lush[1], wet),
        lerp(arid[2], lush[2], wet),
        kOne
    };

    vec4<f64> mid = vec4<f64>{
        lerp(land[0], P.highland[0], low),
        lerp(land[1], P.highland[1], low),
        lerp(land[2], P.highland[2], low),
        kOne
    };

    vec4<f64> rocky = vec4<f64>{
        lerp(mid[0], P.rock[0], high),
        lerp(mid[1], P.rock[1], high),
        lerp(mid[2], P.rock[2], high),
        kOne
    };

    f64 cold = smoothstep(f64{0.55}, f64{0.92}, kOne - temp);
    vec4<f64> snowy = vec4<f64>{
        lerp(rocky[0], P.snow[0], peak * cold),
        lerp(rocky[1], P.snow[1], peak * cold),
        lerp(rocky[2], P.snow[2], peak * cold),
        kOne
    };

    vec4<f64> landmix = vec4<f64>{
        lerp(P.beach[0], snowy[0], shore),
        lerp(P.beach[1], snowy[1], shore),
        lerp(P.beach[2], snowy[2], shore),
        kOne
    };

    vec4<f64> finalc = vec4<f64>{
        lerp(c[0], landmix[0], shore),
        lerp(c[1], landmix[1], shore),
        lerp(c[2], landmix[2], shore),
        kOne
    };

    return finalc;
}

static inline vec4<f64> shade_atmosphere(const planet_descriptor& d, f64 ndotv, f64 ndotl) noexcept {
    auto A = d.palette.atmosphere;
    f64 horizon = std::pow(clamp(kOne - ndotv, kZero, kOne), lerp(f64{1.25}, f64{4.25}, d.atmosphere_height));
    f64 sun = std::pow(clamp(ndotl, kZero, kOne), f64{8.0});
    f64 alpha = clamp(A[3] * (horizon + sun * f64{0.35}), kZero, kOne);
    return vec4<f64>{ A[0], A[1], A[2], alpha };
}

struct image final {
    u32 w{};
    u32 h{};
    std::vector<u32> px;

    image(u32 W, u32 H) : w(W), h(H), px(static_cast<std::size_t>(W) * static_cast<std::size_t>(H), pack_rgba8(vec4<f64>{kZero,kZero,kZero,kOne})) {}

    u32& at(u32 x, u32 y) { return px[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)]; }
    const u32& at(u32 x, u32 y) const { return px[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)]; }
};

static inline std::expected<void, std::string> write_ppm(const std::filesystem::path& path, const image& img) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return std::unexpected(std::string("io.open.failed"));
    out << "P6\n" << img.w << " " << img.h << "\n" << u32{255} << "\n";
    for (u32 y = kZeroU; y < img.h; y += kOneU) {
        for (u32 x = kZeroU; x < img.w; x += kOneU) {
            auto c = unpack_rgba8(img.at(x,y));
            u8 r = static_cast<u8>(clamp(c[0], kZero, kOne) * f64{255.0} + kHalf);
            u8 g = static_cast<u8>(clamp(c[1], kZero, kOne) * f64{255.0} + kHalf);
            u8 b = static_cast<u8>(clamp(c[2], kZero, kOne) * f64{255.0} + kHalf);
            out.write(reinterpret_cast<const char*>(&r), sizeof(r));
            out.write(reinterpret_cast<const char*>(&g), sizeof(g));
            out.write(reinterpret_cast<const char*>(&b), sizeof(b));
        }
    }
    return {};
}

struct ray final {
    vec3<f64> o{};
    vec3<f64> d{};
};

static inline ray camera_ray(const camera& cam, f64 sx, f64 sy) noexcept {
    auto B = cam.view_basis();
    f64 tan_half = std::tan(cam.vfov_rad * kHalf);
    f64 px = (sx * kTwo - kOne) * cam.aspect * tan_half;
    f64 py = (kOne - sy * kTwo) * tan_half;
    auto dir = (B.c0 * px + B.c1 * py + B.c2 * kOne).normalized();
    return ray{ cam.pos, dir };
}

static inline std::optional<std::pair<f64,f64>> intersect_sphere(const ray& r, f64 radius) noexcept {
    f64 a = r.d.dot(r.d);
    f64 b = kTwo * r.o.dot(r.d);
    f64 c = r.o.dot(r.o) - radius * radius;
    f64 disc = b*b - kFour*a*c;
    if (disc < kZero) return std::nullopt;
    f64 s = std::sqrt(static_cast<long double>(disc));
    f64 inv2a = kOne / (kTwo*a + kTiny);
    f64 t0 = (-b - s) * inv2a;
    f64 t1 = (-b + s) * inv2a;
    if (t1 < kZero) return std::nullopt;
    if (t0 < kZero) t0 = t1;
    return std::pair<f64,f64>{t0, t1};
}

static inline vec4<f64> sky(vec3<f64> dir, vec3<f64> sun_dir) noexcept {
    f64 t = clamp(dir[2] * kHalf + kHalf, kZero, kOne);
    vec4<f64> zenith{ f64{0.03}, f64{0.07}, f64{0.12}, kOne };
    vec4<f64> horizon{ f64{0.25}, f64{0.35}, f64{0.55}, kOne };
    vec4<f64> base{
        lerp(horizon[0], zenith[0], s_curve(t)),
        lerp(horizon[1], zenith[1], s_curve(t)),
        lerp(horizon[2], zenith[2], s_curve(t)),
        kOne
    };
    f64 sund = clamp(dir.normalized().dot(sun_dir.normalized()), kZero, kOne);
    f64 glow = std::pow(sund, f64{64.0});
    vec4<f64> sun{ f64{1.0}, f64{0.92}, f64{0.78}, kOne };
    return vec4<f64>{
        clamp(base[0] + sun[0] * glow, kZero, kOne),
        clamp(base[1] + sun[1] * glow, kZero, kOne),
        clamp(base[2] + sun[2] * glow, kZero, kOne),
        kOne
    };
}

struct render_params final {
    u32 width{ u32{1024} };
    u32 height{ u32{1024} };
    f64 exposure{ f64{1.0} };
    f64 gamma{ f64{2.2} };
    f64 atmosphere_boost{ f64{1.0} };
};

static inline vec4<f64> tonemap(vec4<f64> c, const render_params& rp) noexcept {
    vec3<f64> rgb{ c[0]*rp.exposure, c[1]*rp.exposure, c[2]*rp.exposure };
    auto aces = [](f64 x) noexcept {
        const f64 a = f64{2.51};
        const f64 b = f64{0.03};
        const f64 c = f64{2.43};
        const f64 d = f64{0.59};
        const f64 e = f64{0.14};
        return clamp((x*(a*x+b)) / (x*(c*x+d)+e), kZero, kOne);
    };
    rgb = vec3<f64>{ aces(rgb[0]), aces(rgb[1]), aces(rgb[2]) };
    f64 invg = kOne / rp.gamma;
    rgb = vec3<f64>{ std::pow(rgb[0], invg), std::pow(rgb[1], invg), std::pow(rgb[2], invg) };
    return vec4<f64>{ rgb[0], rgb[1], rgb[2], c[3] };
}

static inline image render_planet(const planet_descriptor& d, const camera& cam, render_params rp) {
    image img(rp.width, rp.height);
    auto sun = d.sun_dir.normalized();

    for (u32 y = kZeroU; y < rp.height; y += kOneU) {
        for (u32 x = kZeroU; x < rp.width; x += kOneU) {
            f64 sx = (static_cast<f64>(x) + kHalf) / static_cast<f64>(rp.width);
            f64 sy = (static_cast<f64>(y) + kHalf) / static_cast<f64>(rp.height);

            auto r = camera_ray(cam, sx, sy);
            auto hit = intersect_sphere(r, d.radius);

            vec4<f64> outc = sky(r.d, sun);

            if (hit) {
                f64 t = hit->first;
                auto p = r.o + r.d * t;
                auto n = (p / d.radius).normalized();

                f64 h = terrain_height(n, d);
                f64 temp = climate_temperature(n, d.axial_tilt_rad, sun);
                f64 humid = climate_humidity(n, d.seed);

                vec4<f64> albedo = shade_surface(d, h, temp, humid);

                f64 ndotl = clamp(n.dot(sun), kZero, kOne);
                f64 ndotv = clamp(n.dot(-r.d), kZero, kOne);

                f64 rim = std::pow(clamp(kOne - ndotv, kZero, kOne), f64{2.0});
                f64 diffuse = ndotl;
                f64 ambient = lerp(f64{0.06}, f64{0.22}, temp);

                vec3<f64> lit{
                    albedo[0] * (ambient + diffuse * d.sun_intensity),
                    albedo[1] * (ambient + diffuse * d.sun_intensity),
                    albedo[2] * (ambient + diffuse * d.sun_intensity)
                };

                f64 spec_pow = lerp(f64{32.0}, f64{256.0}, smoothstep(f64{0.15}, f64{0.85}, kOne - humid));
                auto hvec = (sun + (-r.d)).normalized();
                f64 spec = std::pow(clamp(n.dot(hvec), kZero, kOne), spec_pow) * lerp(f64{0.02}, f64{0.18}, (kOne - humid));
                lit += vec3<f64>{spec, spec, spec};

                vec4<f64> atm = shade_atmosphere(d, ndotv, ndotl);
                atm[3] *= rp.atmosphere_boost;

                vec3<f64> atm_rgb{ atm[0]*atm[3], atm[1]*atm[3], atm[2]*atm[3] };
                vec3<f64> rgb = lit * (kOne - atm[3]) + atm_rgb;

                outc = tonemap(vec4<f64>{ rgb[0], rgb[1], rgb[2], kOne }, rp);
            } else {
                outc = tonemap(outc, rp);
            }

            img.at(x,y) = pack_rgba8(outc);
        }
    }

    return img;
}

static inline f64 wrap2pi(f64 x) noexcept {
    f64 t = std::fmod(x, kTau);
    return (t < kZero) ? (t + kTau) : t;
}

static inline f64 solve_kepler(f64 M, f64 e) noexcept {
    M = wrap2pi(M);
    f64 E = (e < f64{0.8}) ? M : kPi;
    const i32 kIter = i32{18};
    const f64 kTol = f64{1e-14};
    for (i32 it = kZeroI; it < kIter; it += kOneI) {
        f64 s = std::sin(E);
        f64 c = std::cos(E);
        f64 f = E - e*s - M;
        f64 fp = kOne - e*c;
        f64 d = f / (fp + kTiny);
        E -= d;
        if (std::abs(d) < kTol) break;
    }
    return E;
}

static inline std::pair<vec3<f64>, vec3<f64>> orbit_state(const orbit_params& o, f64 t) noexcept {
    f64 n = std::sqrt(o.mu_central / (o.semi_major*o.semi_major*o.semi_major + kTiny));
    f64 M = o.mean_anomaly0_rad + n * t;
    f64 E = solve_kepler(M, o.eccentricity);

    f64 cE = std::cos(E);
    f64 sE = std::sin(E);

    f64 one_minus_ecE = (kOne - o.eccentricity*cE);
    f64 r = o.semi_major * one_minus_ecE;

    f64 x = o.semi_major * (cE - o.eccentricity);
    f64 y = o.semi_major * (std::sqrt(clamp(kOne - o.eccentricity*o.eccentricity, kZero, kOne)) * sE);

    f64 vx = -o.semi_major * n * sE / (one_minus_ecE + kTiny);
    f64 vy =  o.semi_major * n * std::sqrt(clamp(kOne - o.eccentricity*o.eccentricity, kZero, kOne)) * cE / (one_minus_ecE + kTiny);

    f64 cO = std::cos(o.asc_node_rad), sO = std::sin(o.asc_node_rad);
    f64 co = std::cos(o.arg_periapsis_rad), so = std::sin(o.arg_periapsis_rad);
    f64 ci = std::cos(o.inclination_rad), si = std::sin(o.inclination_rad);

    f64 R11 = cO*co - sO*so*ci;
    f64 R12 = -cO*so - sO*co*ci;
    f64 R21 = sO*co + cO*so*ci;
    f64 R22 = -sO*so + cO*co*ci;
    f64 R31 = so*si;
    f64 R32 = co*si;

    vec3<f64> rr{ R11*x + R12*y, R21*x + R22*y, R31*x + R32*y };
    vec3<f64> vv{ R11*vx + R12*vy, R21*vx + R22*vy, R31*vx + R32*vy };
    return { rr, vv };
}

static inline orbit_params cool_orbit(u64 seed) noexcept {
    splitmix64 rng(seed ^ u64{0x0RB1TULL});
    orbit_params o{};
    o.semi_major = rng.uniform(f64{0.7}, f64{2.4});
    o.eccentricity = clamp(rng.uniform(f64{0.0}, f64{0.25}), kZero, f64{0.85});
    o.inclination_rad = rng.uniform(kZero, f64{12.0} * kDegToRad);
    o.asc_node_rad = rng.uniform(kZero, kTau);
    o.arg_periapsis_rad = rng.uniform(kZero, kTau);
    o.mean_anomaly0_rad = rng.uniform(kZero, kTau);
    o.mu_central = f64{1.0};
    return o;
}

int main() {
    planet_descriptor pd{};
    pd.seed = u64{0xA11CE5EEDULL};
    pd.radius = f64{6.371e6};
    pd.ocean_level = f64{0.38};
    pd.mountain_amp = f64{0.22};
    pd.continental_amp = f64{0.62};
    pd.roughness = f64{0.90};
    pd.atmosphere_height = f64{0.07};
    pd.axial_tilt_rad = f64{27.0} * kDegToRad;
    pd.sun_intensity = f64{1.35};
    pd.sun_dir = vec3<f64>{ f64{0.37}, f64{0.18}, f64{0.91} }.normalized();

    camera cam{};
    cam.pos = vec3<f64>{ f64{0.0}, -pd.radius * f64{3.2}, pd.radius * f64{0.9} };
    cam.look = vec3<f64>{ kZero, kZero, kZero };
    cam.up = kAxisZ;
    cam.vfov_rad = f64{42.0} * kDegToRad;
    cam.aspect = f64{1.0};

    render_params rp{};
    rp.width = u32{1024};
    rp.height = u32{1024};
    rp.exposure = f64{1.05};
    rp.gamma = f64{2.2};
    rp.atmosphere_boost = f64{1.15};

    auto img = render_planet(pd, cam, rp);

    auto out = write_ppm("planet.ppm", img);
    if (!out) {
        std::cerr << "write failed: " << out.error() << "\n";
        return kOneI;
    }

    orbit_params orb = cool_orbit(pd.seed);
    const u32 kSamples = u32{64};
    const f64 kPeriod = kTau;
    vec3<f64> centroid{ kZero, kZero, kZero };
    for (u32 i = kZeroU; i < kSamples; i += kOneU) {
        f64 t = (static_cast<f64>(i) / static_cast<f64>(kSamples - kOneU)) * kPeriod;
        auto [r,v] = orbit_state(orb, t);
        centroid += r;
        (void)v;
    }
    centroid /= static_cast<f64>(kSamples);

    std::cout << std::string(kFileTag) << " planet.ppm written\n";
    std::cout << "orbit centroid: " << std::fixed << std::setprecision(kThreeI)
              << centroid[0] << " " << centroid[1] << " " << centroid[2] << "\n";

    return kZeroI;
}
```
```cpp
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace planet {

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using f32 = float;
using f64 = double;

static constexpr std::string_view kFileTag = "planet.cpcp";

static constexpr i32 kZeroI = i32{0};
static constexpr i32 kOneI  = i32{1};
static constexpr i32 kTwoI  = i32{2};
static constexpr i32 kThreeI = i32{3};
static constexpr i32 kFourI = i32{4};
static constexpr i32 kFiveI = i32{5};
static constexpr i32 kSixI = i32{6};
static constexpr i32 kSevenI = i32{7};
static constexpr i32 kEightI = i32{8};
static constexpr i32 kNineI = i32{9};

static constexpr u32 kZeroU = u32{0};
static constexpr u32 kOneU  = u32{1};
static constexpr u32 kTwoU  = u32{2};
static constexpr u32 kThreeU = u32{3};
static constexpr u32 kFourU = u32{4};

static constexpr f64 kZero = f64{0.0};
static constexpr f64 kOne  = f64{1.0};
static constexpr f64 kTwo  = f64{2.0};
static constexpr f64 kThree = f64{3.0};
static constexpr f64 kFour = f64{4.0};
static constexpr f64 kHalf = f64{0.5};
static constexpr f64 kQuarter = f64{0.25};
static constexpr f64 kEighth = f64{0.125};

static constexpr f64 kPi  = std::numbers::pi_v<f64>;
static constexpr f64 kTau = kTwo * kPi;
static constexpr f64 kDegToRad = kPi / f64{180.0};
static constexpr f64 kRadToDeg = f64{180.0} / kPi;

static constexpr f64 kEpsilon = std::numeric_limits<f64>::epsilon();
static constexpr f64 kTiny = f64{1e-12};
static constexpr f64 kSafeMin = f64{1e-30};
static constexpr f64 kSafeMax = f64{1e+30};

template<class T>
concept Scalar = std::is_arithmetic_v<T> && (!std::is_same_v<T,bool>);

template<Scalar T, std::size_t N>
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

    T norm() const noexcept {
        return std::sqrt(static_cast<long double>(norm2()));
    }

    vec normalized(T eps = static_cast<T>(kEpsilon)) const noexcept {
        auto n = norm();
        if (n <= eps) return *this;
        return (*this) / static_cast<T>(n);
    }
};

template<Scalar T> using vec2 = vec<T,2>;
template<Scalar T> using vec3 = vec<T,3>;
template<Scalar T> using vec4 = vec<T,4>;

static constexpr vec3<f64> kAxisX{ kOne, kZero, kZero };
static constexpr vec3<f64> kAxisY{ kZero, kOne, kZero };
static constexpr vec3<f64> kAxisZ{ kZero, kZero, kOne };

template<Scalar T>
constexpr vec3<T> cross(const vec3<T>& a, const vec3<T>& b) noexcept {
    return vec3<T>{
        a.v[1]*b.v[2] - a.v[2]*b.v[1],
        a.v[2]*b.v[0] - a.v[0]*b.v[2],
        a.v[0]*b.v[1] - a.v[1]*b.v[0]
    };
}

template<Scalar T>
constexpr T clamp(T x, T lo, T hi) noexcept {
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}

template<Scalar T>
constexpr T lerp(T a, T b, T t) noexcept {
    return a + (b - a) * t;
}

constexpr u64 mix64(u64 x) noexcept {
    x ^= (x >> u64{33});
    x *= u64{0xff51afd7ed558ccdULL};
    x ^= (x >> u64{33});
    x *= u64{0xc4ceb9fe1a85ec53ULL};
    x ^= (x >> u64{33});
    return x;
}

struct splitmix64 final {
    u64 x{};
    constexpr explicit splitmix64(u64 seed) : x(seed) {}

    constexpr u64 next_u64() noexcept {
        x += u64{0x9e3779b97f4a7c15ULL};
        u64 z = x;
        z = (z ^ (z >> u64{30})) * u64{0xbf58476d1ce4e5b9ULL};
        z = (z ^ (z >> u64{27})) * u64{0x94d049bb133111ebULL};
        return z ^ (z >> u64{31});
    }

    f64 uniform01() noexcept {
        constexpr u64 kMantBits = u64{53};
        constexpr u64 kShift = u64{64} - kMantBits;
        u64 u = next_u64() >> kShift;
        f64 denom = static_cast<f64>(u64{1} << kMantBits);
        return static_cast<f64>(u) / denom;
    }

    f64 uniform(f64 lo, f64 hi) noexcept {
        return lerp(lo, hi, uniform01());
    }
};

constexpr u32 pack_rgba8(vec4<f64> c) noexcept {
    auto to8 = [](f64 x) constexpr -> u32 {
        f64 y = clamp(x, kZero, kOne);
        f64 scaled = y * static_cast<f64>(u32{255});
        return static_cast<u32>(scaled + kHalf);
    };
    u32 r = to8(c[0]);
    u32 g = to8(c[1]);
    u32 b = to8(c[2]);
    u32 a = to8(c[3]);
    return (a << u32{24}) | (b << u32{16}) | (g << u32{8}) | r;
}

constexpr vec4<f64> unpack_rgba8(u32 p) noexcept {
    auto f = [](u32 x) constexpr -> f64 { return static_cast<f64>(x) / static_cast<f64>(u32{255}); };
    u32 r = (p >> u32{0})  & u32{255};
    u32 g = (p >> u32{8})  & u32{255};
    u32 b = (p >> u32{16}) & u32{255};
    u32 a = (p >> u32{24}) & u32{255};
    return vec4<f64>{ f(r), f(g), f(b), f(a) };
}

struct mat3 final {
    vec3<f64> c0{ kOne, kZero, kZero };
    vec3<f64> c1{ kZero, kOne, kZero };
    vec3<f64> c2{ kZero, kZero, kOne };

    static mat3 from_cols(vec3<f64> a, vec3<f64> b, vec3<f64> c) noexcept {
        mat3 m;
        m.c0 = a; m.c1 = b; m.c2 = c;
        return m;
    }

    vec3<f64> mul(vec3<f64> v) const noexcept {
        return c0 * v[0] + c1 * v[1] + c2 * v[2];
    }
};

mat3 basis_from_forward(vec3<f64> fwd, vec3<f64> up_hint) noexcept {
    auto f = fwd.normalized();
    auto r = cross(up_hint, f);
    if (r.norm2() <= kTiny) r = cross(kAxisY, f);
    r = r.normalized();
    auto u = cross(f, r).normalized();
    return mat3::from_cols(r, u, f);
}

struct camera final {
    vec3<f64> pos{ kZero, kZero, kZero };
    vec3<f64> look{ kZero, kZero, kOne };
    vec3<f64> up{ kZero, kOne, kZero };
    f64 vfov_rad{ f64{60.0} * kDegToRad };
    f64 aspect{ f64{16.0} / f64{9.0} };

    mat3 view_basis() const noexcept {
        return basis_from_forward(look - pos, up);
    }
};

struct surface_palette final {
    vec4<f64> deep_ocean{ f64{0.02}, f64{0.08}, f64{0.18}, kOne };
    vec4<f64> shallow_ocean{ f64{0.02}, f64{0.18}, f64{0.24}, kOne };
    vec4<f64> beach{ f64{0.76}, f64{0.70}, f64{0.52}, kOne };
    vec4<f64> lowland{ f64{0.10}, f64{0.35}, f64{0.18}, kOne };
    vec4<f64> highland{ f64{0.22}, f64{0.40}, f64{0.26}, kOne };
    vec4<f64> rock{ f64{0.42}, f64{0.42}, f64{0.44}, kOne };
    vec4<f64> snow{ f64{0.92}, f64{0.94}, f64{0.98}, kOne };
    vec4<f64> atmosphere{ f64{0.30}, f64{0.60}, f64{0.95}, f64{0.65} };
};

struct planet_descriptor final {
    u64 seed{ u64{0xC0FFEEULL} };
    f64 radius{ f64{6.371e6} };
    f64 ocean_level{ f64{0.35} };
    f64 mountain_amp{ f64{0.18} };
    f64 continental_amp{ f64{0.55} };
    f64 roughness{ f64{0.85} };
    f64 atmosphere_height{ f64{0.06} };
    f64 axial_tilt_rad{ f64{23.44} * kDegToRad };
    f64 sun_intensity{ f64{1.25} };
    vec3<f64> sun_dir{ f64{0.4}, f64{0.3}, f64{0.86} };
    surface_palette palette{};
};

struct orbit_params final {
    f64 semi_major{ f64{1.0} };
    f64 eccentricity{ f64{0.0} };
    f64 inclination_rad{ kZero };
    f64 asc_node_rad{ kZero };
    f64 arg_periapsis_rad{ kZero };
    f64 mean_anomaly0_rad{ kZero };
    f64 mu_central{ f64{1.0} };
};

struct vertex final {
    vec3<f64> pos{};
    vec3<f64> nrm{};
    vec2<f64> uv{};
    vec4<f64> albedo{};
    f64 height{};
    f64 humidity{};
    f64 temperature{};
};

struct mesh final {
    std::vector<vertex> vtx;
    std::vector<u32> idx;
};

constexpr f64 smoothstep(f64 e0, f64 e1, f64 x) noexcept {
    f64 t = clamp((x - e0) / (e1 - e0 + kTiny), kZero, kOne);
    return t * t * (kThree - kTwo * t);
}

constexpr f64 remap01(f64 x, f64 lo, f64 hi) noexcept {
    return clamp((x - lo) / (hi - lo + kTiny), kZero, kOne);
}

constexpr f64 s_curve(f64 t) noexcept {
    return t * t * (kThree - kTwo * t);
}

static inline u64 hash_grid(i64 x, i64 y, i64 z, u64 seed) noexcept {
    u64 h = seed;
    h ^= mix64(static_cast<u64>(x) + u64{0x9e3779b97f4a7c15ULL});
    h ^= mix64(static_cast<u64>(y) + u64{0xbf58476d1ce4e5b9ULL});
    h ^= mix64(static_cast<u64>(z) + u64{0x94d049bb133111ebULL});
    return mix64(h);
}

static inline f64 grad(u64 h, f64 x, f64 y, f64 z) noexcept {
    constexpr u64 kMask = u64{15};
    u64 g = h & kMask;
    f64 u = (g < u64{8}) ? x : y;
    f64 v = (g < u64{4}) ? y : ((g == u64{12} || g == u64{14}) ? x : z);
    f64 s0 = ((g & u64{1}) == u64{0}) ? u : -u;
    f64 s1 = ((g & u64{2}) == u64{0}) ? v : -v;
    return s0 + s1;
}

static inline f64 fade(f64 t) noexcept {
    return t * t * t * (t * (t * f64{6.0} - f64{15.0}) + f64{10.0});
}

static inline f64 noise3(vec3<f64> p, u64 seed) noexcept {
    f64 fx = std::floor(p[0]);
    f64 fy = std::floor(p[1]);
    f64 fz = std::floor(p[2]);

    i64 X = static_cast<i64>(fx);
    i64 Y = static_cast<i64>(fy);
    i64 Z = static_cast<i64>(fz);

    f64 x = p[0] - fx;
    f64 y = p[1] - fy;
    f64 z = p[2] - fz;

    f64 u = fade(x);
    f64 v = fade(y);
    f64 w = fade(z);

    auto H = [&](i64 xi, i64 yi, i64 zi) -> u64 { return hash_grid(xi, yi, zi, seed); };

    f64 n000 = grad(H(X,   Y,   Z  ), x,     y,     z    );
    f64 n100 = grad(H(X+1, Y,   Z  ), x-kOne, y,     z    );
    f64 n010 = grad(H(X,   Y+1, Z  ), x,     y-kOne, z    );
    f64 n110 = grad(H(X+1, Y+1, Z  ), x-kOne, y-kOne, z   );
    f64 n001 = grad(H(X,   Y,   Z+1), x,     y,     z-kOne);
    f64 n101 = grad(H(X+1, Y,   Z+1), x-kOne, y,     z-kOne);
    f64 n011 = grad(H(X,   Y+1, Z+1), x,     y-kOne, z-kOne);
    f64 n111 = grad(H(X+1, Y+1, Z+1), x-kOne, y-kOne, z-kOne);

    f64 nx00 = lerp(n000, n100, u);
    f64 nx10 = lerp(n010, n110, u);
    f64 nx01 = lerp(n001, n101, u);
    f64 nx11 = lerp(n011, n111, u);

    f64 nxy0 = lerp(nx00, nx10, v);
    f64 nxy1 = lerp(nx01, nx11, v);

    f64 nxyz = lerp(nxy0, nxy1, w);
    return nxyz;
}

struct fbm_params final {
    u32 octaves{ u32{7} };
    f64 lacunarity{ f64{2.0} };
    f64 gain{ f64{0.5} };
    f64 base_freq{ f64{1.0} };
    f64 base_amp{ f64{1.0} };
};

static inline f64 fbm(vec3<f64> p, u64 seed, fbm_params fp) noexcept {
    f64 sum = kZero;
    f64 amp = fp.base_amp;
    f64 freq = fp.base_freq;
    for (u32 i = kZeroU; i < fp.octaves; i += kOneU) {
        f64 n = noise3(p * freq, seed ^ mix64(static_cast<u64>(i)));
        sum += n * amp;
        freq *= fp.lacunarity;
        amp *= fp.gain;
    }
    return sum;
}

static inline vec3<f64> spherify(vec3<f64> p) noexcept {
    return p.normalized();
}

static inline vec2<f64> sphere_uv(vec3<f64> n) noexcept {
    f64 u = (std::atan2(n[1], n[0]) + kPi) / kTau;
    f64 v = (std::asin(clamp(n[2], -kOne, kOne)) + kPi * kHalf) / kPi;
    return vec2<f64>{u, v};
}

static inline vec3<f64> rotate_axis_angle(vec3<f64> v, vec3<f64> axis, f64 angle) noexcept {
    auto a = axis.normalized();
    f64 c = std::cos(angle);
    f64 s = std::sin(angle);
    return v * c + cross(a, v) * s + a * (a.dot(v) * (kOne - c));
}

static inline f64 climate_temperature(vec3<f64> n, f64 axial_tilt_rad, vec3<f64> sun_dir) noexcept {
    auto tilt_axis = rotate_axis_angle(kAxisY, kAxisX, axial_tilt_rad).normalized();
    auto equator = cross(tilt_axis, kAxisZ).normalized();
    f64 lat = std::asin(clamp(n.dot(tilt_axis), -kOne, kOne));
    f64 latitude01 = kOne - std::abs(lat) / (kPi * kHalf);
    f64 insolation = clamp(n.dot(sun_dir.normalized()), kZero, kOne);
    f64 t = s_curve(latitude01) * lerp(f64{0.35}, f64{1.0}, insolation);
    return clamp(t, kZero, kOne);
}

static inline f64 climate_humidity(vec3<f64> n, u64 seed) noexcept {
    fbm_params fp{};
    fp.octaves = u32{6};
    fp.lacunarity = f64{2.15};
    fp.gain = f64{0.55};
    fp.base_freq = f64{1.75};
    fp.base_amp = f64{1.0};
    f64 h = fbm(n * f64{3.0}, seed ^ u64{0xA11CEULL}, fp);
    f64 r = remap01(h, -f64{0.9}, f64{0.9});
    return clamp(r, kZero, kOne);
}

static inline f64 terrain_height(vec3<f64> n, const planet_descriptor& d) noexcept {
    fbm_params continents{};
    continents.octaves = u32{5};
    continents.lacunarity = f64{2.0};
    continents.gain = f64{0.5};
    continents.base_freq = f64{0.85};
    continents.base_amp = f64{1.0};

    fbm_params mountains{};
    mountains.octaves = u32{8};
    mountains.lacunarity = f64{2.25};
    mountains.gain = f64{0.48};
    mountains.base_freq = f64{3.25};
    mountains.base_amp = f64{1.0};

    f64 c = fbm(n, d.seed ^ u64{0xC0NT1N3NTULL}, continents);
    f64 m = fbm(n, d.seed ^ u64{0xM0UNT41NULL}, mountains);

    f64 c01 = remap01(c, -f64{0.8}, f64{0.8});
    f64 ridges = kOne - std::abs(m);
    f64 ridgy = std::pow(clamp(ridges, kZero, kOne), lerp(f64{1.25}, f64{3.75}, d.roughness));

    f64 base = (c01 - d.ocean_level) * d.continental_amp;
    f64 peaks = (ridgy - f64{0.45}) * d.mountain_amp;
    f64 h = base + peaks;

    f64 cap = f64{0.95};
    h = clamp(h, -cap, cap);
    return h;
}

static inline vec4<f64> shade_surface(const planet_descriptor& d, f64 h, f64 temp, f64 humid) noexcept {
    auto& P = d.palette;
    f64 ocean = smoothstep(-f64{0.65}, -f64{0.05}, h);
    f64 shore = smoothstep(-f64{0.05}, f64{0.02}, h);
    f64 low = smoothstep(f64{0.02}, f64{0.18}, h);
    f64 high = smoothstep(f64{0.18}, f64{0.42}, h);
    f64 peak = smoothstep(f64{0.42}, f64{0.70}, h);

    vec4<f64> c = P.deep_ocean;
    c = vec4<f64>{
        lerp(c[0], P.shallow_ocean[0], ocean),
        lerp(c[1], P.shallow_ocean[1], ocean),
        lerp(c[2], P.shallow_ocean[2], ocean),
        kOne
    };

    vec4<f64> land = P.lowland;
    vec4<f64> lush = vec4<f64>{ P.lowland[0]*f64{0.85}, P.lowland[1]*f64{1.10}, P.lowland[2]*f64{0.80}, kOne };
    vec4<f64> arid = vec4<f64>{ P.lowland[0]*f64{1.10}, P.lowland[1]*f64{0.95}, P.lowland[2]*f64{0.70}, kOne };
    f64 wet = smoothstep(f64{0.35}, f64{0.85}, humid);
    land = vec4<f64>{
        lerp(arid[0], lush[0], wet),
        lerp(arid[1], lush[1], wet),
        lerp(arid[2], lush[2], wet),
        kOne
    };

    vec4<f64> mid = vec4<f64>{
        lerp(land[0], P.highland[0], low),
        lerp(land[1], P.highland[1], low),
        lerp(land[2], P.highland[2], low),
        kOne
    };

    vec4<f64> rocky = vec4<f64>{
        lerp(mid[0], P.rock[0], high),
        lerp(mid[1], P.rock[1], high),
        lerp(mid[2], P.rock[2], high),
        kOne
    };

    f64 cold = smoothstep(f64{0.55}, f64{0.92}, kOne - temp);
    vec4<f64> snowy = vec4<f64>{
        lerp(rocky[0], P.snow[0], peak * cold),
        lerp(rocky[1], P.snow[1], peak * cold),
        lerp(rocky[2], P.snow[2], peak * cold),
        kOne
    };

    vec4<f64> landmix = vec4<f64>{
        lerp(P.beach[0], snowy[0], shore),
        lerp(P.beach[1], snowy[1], shore),
        lerp(P.beach[2], snowy[2], shore),
        kOne
    };

    vec4<f64> finalc = vec4<f64>{
        lerp(c[0], landmix[0], shore),
        lerp(c[1], landmix[1], shore),
        lerp(c[2], landmix[2], shore),
        kOne
    };

    return finalc;
}

static inline vec4<f64> shade_atmosphere(const planet_descriptor& d, f64 ndotv, f64 ndotl) noexcept {
    auto A = d.palette.atmosphere;
    f64 horizon = std::pow(clamp(kOne - ndotv, kZero, kOne), lerp(f64{1.25}, f64{4.25}, d.atmosphere_height));
    f64 sun = std::pow(clamp(ndotl, kZero, kOne), f64{8.0});
    f64 alpha = clamp(A[3] * (horizon + sun * f64{0.35}), kZero, kOne);
    return vec4<f64>{ A[0], A[1], A[2], alpha };
}

struct image final {
    u32 w{};
    u32 h{};
    std::vector<u32> px;

    image(u32 W, u32 H) : w(W), h(H), px(static_cast<std::size_t>(W) * static_cast<std::size_t>(H), pack_rgba8(vec4<f64>{kZero,kZero,kZero,kOne})) {}

    u32& at(u32 x, u32 y) { return px[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)]; }
    const u32& at(u32 x, u32 y) const { return px[static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x)]; }
};

static inline std::expected<void, std::string> write_ppm(const std::filesystem::path& path, const image& img) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return std::unexpected(std::string("io.open.failed"));
    out << "P6\n" << img.w << " " << img.h << "\n" << u32{255} << "\n";
    for (u32 y = kZeroU; y < img.h; y += kOneU) {
        for (u32 x = kZeroU; x < img.w; x += kOneU) {
            auto c = unpack_rgba8(img.at(x,y));
            u8 r = static_cast<u8>(clamp(c[0], kZero, kOne) * f64{255.0} + kHalf);
            u8 g = static_cast<u8>(clamp(c[1], kZero, kOne) * f64{255.0} + kHalf);
            u8 b = static_cast<u8>(clamp(c[2], kZero, kOne) * f64{255.0} + kHalf);
            out.write(reinterpret_cast<const char*>(&r), sizeof(r));
            out.write(reinterpret_cast<const char*>(&g), sizeof(g));
            out.write(reinterpret_cast<const char*>(&b), sizeof(b));
        }
    }
    return {};
}

struct ray final {
    vec3<f64> o{};
    vec3<f64> d{};
};

static inline ray camera_ray(const camera& cam, f64 sx, f64 sy) noexcept {
    auto B = cam.view_basis();
    f64 tan_half = std::tan(cam.vfov_rad * kHalf);
    f64 px = (sx * kTwo - kOne) * cam.aspect * tan_half;
    f64 py = (kOne - sy * kTwo) * tan_half;
    auto dir = (B.c0 * px + B.c1 * py + B.c2 * kOne).normalized();
    return ray{ cam.pos, dir };
}

static inline std::optional<std::pair<f64,f64>> intersect_sphere(const ray& r, f64 radius) noexcept {
    f64 a = r.d.dot(r.d);
    f64 b = kTwo * r.o.dot(r.d);
    f64 c = r.o.dot(r.o) - radius * radius;
    f64 disc = b*b - kFour*a*c;
    if (disc < kZero) return std::nullopt;
    f64 s = std::sqrt(static_cast<long double>(disc));
    f64 inv2a = kOne / (kTwo*a + kTiny);
    f64 t0 = (-b - s) * inv2a;
    f64 t1 = (-b + s) * inv2a;
    if (t1 < kZero) return std::nullopt;
    if (t0 < kZero) t0 = t1;
    return std::pair<f64,f64>{t0, t1};
}

static inline vec4<f64> sky(vec3<f64> dir, vec3<f64> sun_dir) noexcept {
    f64 t = clamp(dir[2] * kHalf + kHalf, kZero, kOne);
    vec4<f64> zenith{ f64{0.03}, f64{0.07}, f64{0.12}, kOne };
    vec4<f64> horizon{ f64{0.25}, f64{0.35}, f64{0.55}, kOne };
    vec4<f64> base{
        lerp(horizon[0], zenith[0], s_curve(t)),
        lerp(horizon[1], zenith[1], s_curve(t)),
        lerp(horizon[2], zenith[2], s_curve(t)),
        kOne
    };
    f64 sund = clamp(dir.normalized().dot(sun_dir.normalized()), kZero, kOne);
    f64 glow = std::pow(sund, f64{64.0});
    vec4<f64> sun{ f64{1.0}, f64{0.92}, f64{0.78}, kOne };
    return vec4<f64>{
        clamp(base[0] + sun[0] * glow, kZero, kOne),
        clamp(base[1] + sun[1] * glow, kZero, kOne),
        clamp(base[2] + sun[2] * glow, kZero, kOne),
        kOne
    };
}

struct render_params final {
    u32 width{ u32{1024} };
    u32 height{ u32{1024} };
    f64 exposure{ f64{1.0} };
    f64 gamma{ f64{2.2} };
    f64 atmosphere_boost{ f64{1.0} };
};

static inline vec4<f64> tonemap(vec4<f64> c, const render_params& rp) noexcept {
    vec3<f64> rgb{ c[0]*rp.exposure, c[1]*rp.exposure, c[2]*rp.exposure };
    auto aces = [](f64 x) noexcept {
        const f64 a = f64{2.51};
        const f64 b = f64{0.03};
        const f64 c = f64{2.43};
        const f64 d = f64{0.59};
        const f64 e = f64{0.14};
        return clamp((x*(a*x+b)) / (x*(c*x+d)+e), kZero, kOne);
    };
    rgb = vec3<f64>{ aces(rgb[0]), aces(rgb[1]), aces(rgb[2]) };
    f64 invg = kOne / rp.gamma;
    rgb = vec3<f64>{ std::pow(rgb[0], invg), std::pow(rgb[1], invg), std::pow(rgb[2], invg) };
    return vec4<f64>{ rgb[0], rgb[1], rgb[2], c[3] };
}

static inline image render_planet(const planet_descriptor& d, const camera& cam, render_params rp) {
    image img(rp.width, rp.height);
    auto sun = d.sun_dir.normalized();

    for (u32 y = kZeroU; y < rp.height; y += kOneU) {
        for (u32 x = kZeroU; x < rp.width; x += kOneU) {
            f64 sx = (static_cast<f64>(x) + kHalf) / static_cast<f64>(rp.width);
            f64 sy = (static_cast<f64>(y) + kHalf) / static_cast<f64>(rp.height);

            auto r = camera_ray(cam, sx, sy);
            auto hit = intersect_sphere(r, d.radius);

            vec4<f64> outc = sky(r.d, sun);

            if (hit) {
                f64 t = hit->first;
                auto p = r.o + r.d * t;
                auto n = (p / d.radius).normalized();

                f64 h = terrain_height(n, d);
                f64 temp = climate_temperature(n, d.axial_tilt_rad, sun);
                f64 humid = climate_humidity(n, d.seed);

                vec4<f64> albedo = shade_surface(d, h, temp, humid);

                f64 ndotl = clamp(n.dot(sun), kZero, kOne);
                f64 ndotv = clamp(n.dot(-r.d), kZero, kOne);

                f64 rim = std::pow(clamp(kOne - ndotv, kZero, kOne), f64{2.0});
                f64 diffuse = ndotl;
                f64 ambient = lerp(f64{0.06}, f64{0.22}, temp);

                vec3<f64> lit{
                    albedo[0] * (ambient + diffuse * d.sun_intensity),
                    albedo[1] * (ambient + diffuse * d.sun_intensity),
                    albedo[2] * (ambient + diffuse * d.sun_intensity)
                };

                f64 spec_pow = lerp(f64{32.0}, f64{256.0}, smoothstep(f64{0.15}, f64{0.85}, kOne - humid));
                auto hvec = (sun + (-r.d)).normalized();
                f64 spec = std::pow(clamp(n.dot(hvec), kZero, kOne), spec_pow) * lerp(f64{0.02}, f64{0.18}, (kOne - humid));
                lit += vec3<f64>{spec, spec, spec};

                vec4<f64> atm = shade_atmosphere(d, ndotv, ndotl);
                atm[3] *= rp.atmosphere_boost;

                vec3<f64> atm_rgb{ atm[0]*atm[3], atm[1]*atm[3], atm[2]*atm[3] };
                vec3<f64> rgb = lit * (kOne - atm[3]) + atm_rgb;

                outc = tonemap(vec4<f64>{ rgb[0], rgb[1], rgb[2], kOne }, rp);
            } else {
                outc = tonemap(outc, rp);
            }

            img.at(x,y) = pack_rgba8(outc);
        }
    }

    return img;
}

static inline f64 wrap2pi(f64 x) noexcept {
    f64 t = std::fmod(x, kTau);
    return (t < kZero) ? (t + kTau) : t;
}

static inline f64 solve_kepler(f64 M, f64 e) noexcept {
    M = wrap2pi(M);
    f64 E = (e < f64{0.8}) ? M : kPi;
    const i32 kIter = i32{18};
    const f64 kTol = f64{1e-14};
    for (i32 it = kZeroI; it < kIter; it += kOneI) {
        f64 s = std::sin(E);
        f64 c = std::cos(E);
        f64 f = E - e*s - M;
        f64 fp = kOne - e*c;
        f64 d = f / (fp + kTiny);
        E -= d;
        if (std::abs(d) < kTol) break;
    }
    return E;
}

static inline std::pair<vec3<f64>, vec3<f64>> orbit_state(const orbit_params& o, f64 t) noexcept {
    f64 n = std::sqrt(o.mu_central / (o.semi_major*o.semi_major*o.semi_major + kTiny));
    f64 M = o.mean_anomaly0_rad + n * t;
    f64 E = solve_kepler(M, o.eccentricity);

    f64 cE = std::cos(E);
    f64 sE = std::sin(E);

    f64 one_minus_ecE = (kOne - o.eccentricity*cE);
    f64 r = o.semi_major * one_minus_ecE;

    f64 x = o.semi_major * (cE - o.eccentricity);
    f64 y = o.semi_major * (std::sqrt(clamp(kOne - o.eccentricity*o.eccentricity, kZero, kOne)) * sE);

    f64 vx = -o.semi_major * n * sE / (one_minus_ecE + kTiny);
    f64 vy =  o.semi_major * n * std::sqrt(clamp(kOne - o.eccentricity*o.eccentricity, kZero, kOne)) * cE / (one_minus_ecE + kTiny);

    f64 cO = std::cos(o.asc_node_rad), sO = std::sin(o.asc_node_rad);
    f64 co = std::cos(o.arg_periapsis_rad), so = std::sin(o.arg_periapsis_rad);
    f64 ci = std::cos(o.inclination_rad), si = std::sin(o.inclination_rad);

    f64 R11 = cO*co - sO*so*ci;
    f64 R12 = -cO*so - sO*co*ci;
    f64 R21 = sO*co + cO*so*ci;
    f64 R22 = -sO*so + cO*co*ci;
    f64 R31 = so*si;
    f64 R32 = co*si;

    vec3<f64> rr{ R11*x + R12*y, R21*x + R22*y, R31*x + R32*y };
    vec3<f64> vv{ R11*vx + R12*vy, R21*vx + R22*vy, R31*vx + R32*vy };
    return { rr, vv };
}

static inline orbit_params cool_orbit(u64 seed) noexcept {
    splitmix64 rng(seed ^ u64{0x0RB1TULL});
    orbit_params o{};
    o.semi_major = rng.uniform(f64{0.7}, f64{2.4});
    o.eccentricity = clamp(rng.uniform(f64{0.0}, f64{0.25}), kZero, f64{0.85});
    o.inclination_rad = rng.uniform(kZero, f64{12.0} * kDegToRad);
    o.asc_node_rad = rng.uniform(kZero, kTau);
    o.arg_periapsis_rad = rng.uniform(kZero, kTau);
    o.mean_anomaly0_rad = rng.uniform(kZero, kTau);
    o.mu_central = f64{1.0};
    return o;
}

int main() {
    planet_descriptor pd{};
    pd.seed = u64{0xA11CE5EEDULL};
    pd.radius = f64{6.371e6};
    pd.ocean_level = f64{0.38};
    pd.mountain_amp = f64{0.22};
    pd.continental_amp = f64{0.62};
    pd.roughness = f64{0.90};
    pd.atmosphere_height = f64{0.07};
    pd.axial_tilt_rad = f64{27.0} * kDegToRad;
    pd.sun_intensity = f64{1.35};
    pd.sun_dir = vec3<f64>{ f64{0.37}, f64{0.18}, f64{0.91} }.normalized();

    camera cam{};
    cam.pos = vec3<f64>{ f64{0.0}, -pd.radius * f64{3.2}, pd.radius * f64{0.9} };
    cam.look = vec3<f64>{ kZero, kZero, kZero };
    cam.up = kAxisZ;
    cam.vfov_rad = f64{42.0} * kDegToRad;
    cam.aspect = f64{1.0};

    render_params rp{};
    rp.width = u32{1024};
    rp.height = u32{1024};
    rp.exposure = f64{1.05};
    rp.gamma = f64{2.2};
    rp.atmosphere_boost = f64{1.15};

    auto img = render_planet(pd, cam, rp);

    auto out = write_ppm("planet.ppm", img);
    if (!out) {
        std::cerr << "write failed: " << out.error() << "\n";
        return kOneI;
    }

    orbit_params orb = cool_orbit(pd.seed);
    const u32 kSamples = u32{64};
    const f64 kPeriod = kTau;
    vec3<f64> centroid{ kZero, kZero, kZero };
    for (u32 i = kZeroU; i < kSamples; i += kOneU) {
        f64 t = (static_cast<f64>(i) / static_cast<f64>(kSamples - kOneU)) * kPeriod;
        auto [r,v] = orbit_state(orb, t);
        centroid += r;
        (void)v;
    }
    centroid /= static_cast<f64>(kSamples);

    std::cout << std::string(kFileTag) << " planet.ppm written\n";
    std::cout << "orbit centroid: " << std::fixed << std::setprecision(kThreeI)
              << centroid[0] << " " << centroid[1] << " " << centroid[2] << "\n";

    return kZeroI;
}
```


#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <compare>
#include <concepts>
#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <queue>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace eng {

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using f32 = float;
using f64 = double;

template<class... Ts>
struct overload : Ts... { using Ts::operator()...; };
template<class... Ts> overload(Ts...) -> overload<Ts...>;

constexpr u64 fnv1a64(std::string_view s) noexcept {
    u64 h = 14695981039346656037ull;
    for (unsigned char c : s) { h ^= u64(c); h *= 1099511628211ull; }
    return h;
}
constexpr u64 mix64(u64 x) noexcept {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return x;
}
constexpr u64 rotl64(u64 x, int r) noexcept { return std::rotl(x, r); }

template<class T>
concept scalar = std::is_arithmetic_v<T> && (!std::is_same_v<T,bool>);

template<scalar T, std::size_t N>
struct vec final {
    std::array<T,N> v{};

    constexpr vec() = default;

    template<class... U>
    requires (sizeof...(U) == N) && (std::convertible_to<U,T> && ...)
    constexpr explicit vec(U&&... u) : v{static_cast<T>(std::forward<U>(u))...} {}

    constexpr T& operator[](std::size_t i) noexcept { return v[i]; }
    constexpr const T& operator[](std::size_t i) const noexcept { return v[i]; }
    constexpr auto operator<=>(const vec&) const = default;

    constexpr vec operator+() const noexcept { return *this; }
    constexpr vec operator-() const noexcept { vec r; for (std::size_t i=0;i<N;++i) r.v[i] = -v[i]; return r; }

    constexpr vec& operator+=(const vec& o) noexcept { for (std::size_t i=0;i<N;++i) v[i] += o.v[i]; return *this; }
    constexpr vec& operator-=(const vec& o) noexcept { for (std::size_t i=0;i<N;++i) v[i] -= o.v[i]; return *this; }
    constexpr vec& operator*=(T s) noexcept { for (std::size_t i=0;i<N;++i) v[i] *= s; return *this; }
    constexpr vec& operator/=(T s) noexcept { for (std::size_t i=0;i<N;++i) v[i] /= s; return *this; }

    friend constexpr vec operator+(vec a, const vec& b) noexcept { return a += b; }
    friend constexpr vec operator-(vec a, const vec& b) noexcept { return a -= b; }
    friend constexpr vec operator*(vec a, T s) noexcept { return a *= s; }
    friend constexpr vec operator*(T s, vec a) noexcept { return a *= s; }
    friend constexpr vec operator/(vec a, T s) noexcept { return a /= s; }

    constexpr T dot(const vec& o) const noexcept { T r{}; for (std::size_t i=0;i<N;++i) r += v[i]*o.v[i]; return r; }
    constexpr T norm2() const noexcept { return dot(*this); }
    T norm() const noexcept { return std::sqrt((long double)norm2()); }

    vec normalized(T eps = std::numeric_limits<T>::epsilon()) const noexcept {
        auto n = norm();
        if (n <= eps) return *this;
        return (*this) / static_cast<T>(n);
    }
};

template<scalar T> using vec2 = vec<T,2>;
template<scalar T> using vec3 = vec<T,3>;
template<scalar T> using vec4 = vec<T,4>;

template<scalar T>
constexpr vec3<T> cross(const vec3<T>& a, const vec3<T>& b) noexcept {
    return vec3<T>{
        a.v[1]*b.v[2] - a.v[2]*b.v[1],
        a.v[2]*b.v[0] - a.v[0]*b.v[2],
        a.v[0]*b.v[1] - a.v[1]*b.v[0]
    };
}

struct splitmix64 final {
    u64 x{};
    constexpr explicit splitmix64(u64 seed = 0x9e3779b97f4a7c15ull) : x(seed) {}
    constexpr u64 next_u64() noexcept {
        u64 z = (x += 0x9e3779b97f4a7c15ull);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
        return z ^ (z >> 31);
    }
    template<std::floating_point T>
    T uniform01() noexcept {
        constexpr int bits = std::numeric_limits<u64>::digits;
        auto u = next_u64();
        auto mant = u >> (bits - 53);
        auto d = (long double)mant / (long double)(1ull<<53);
        return (T)d;
    }
    template<std::integral I>
    I uniform(I lo, I hi) noexcept {
        u64 span = (u64)(hi - lo);
        if (span == 0) return lo;
        u64 r = next_u64();
        u64 m = (u64)((__uint128_t(r) * __uint128_t(span + 1)) >> 64);
        return (I)(lo + (I)m);
    }
};

struct arena final {
    std::byte* base{};
    std::size_t cap{};
    std::size_t head{};
    explicit arena(std::size_t bytes)
        : base((std::byte*)std::aligned_alloc(64, (bytes+63)&~std::size_t(63)))
        , cap(bytes), head(0) {
        if (!base) throw std::bad_alloc{};
    }
    arena(const arena&) = delete;
    arena& operator=(const arena&) = delete;
    arena(arena&& o) noexcept : base(std::exchange(o.base,nullptr)), cap(std::exchange(o.cap,0)), head(std::exchange(o.head,0)) {}
    arena& operator=(arena&& o) noexcept {
        if (this==&o) return *this;
        if (base) std::free(base);
        base = std::exchange(o.base,nullptr);
        cap  = std::exchange(o.cap,0);
        head = std::exchange(o.head,0);
        return *this;
    }
    ~arena() { if (base) std::free(base); }
    void reset() noexcept { head = 0; }
    void* alloc(std::size_t bytes, std::size_t align = alignof(std::max_align_t)) {
        auto p = (std::uintptr_t)base + head;
        auto aligned = (p + (align-1)) & ~(align-1);
        auto next = (aligned - (std::uintptr_t)base) + bytes;
        if (next > cap) throw std::bad_alloc{};
        head = next;
        return (void*)aligned;
    }
    template<class T, class... A>
    T* make(A&&... a) {
        void* p = alloc(sizeof(T), alignof(T));
        return std::construct_at((T*)p, std::forward<A>(a)...);
    }
};

class scheduler final {
    struct q final {
        std::mutex m;
        std::deque<std::function<void()>> d;
        void push(std::function<void()> f){ std::lock_guard lg(m); d.emplace_back(std::move(f)); }
        std::optional<std::function<void()>> pop_front(){
            std::lock_guard lg(m);
            if (d.empty()) return std::nullopt;
            auto f = std::move(d.front());
            d.pop_front();
            return f;
        }
        std::optional<std::function<void()>> steal_back(){
            std::lock_guard lg(m);
            if (d.empty()) return std::nullopt;
            auto f = std::move(d.back());
            d.pop_back();
            return f;
        }
    };

    std::vector<std::thread> w;
    std::vector<std::unique_ptr<q>> qs;
    std::atomic<u64> inflight{0};
    std::
