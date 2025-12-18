#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "engine/vec.hpp"
#include "engine/rng.hpp"

namespace eng::sim {

using f64 = double;
using v3 = eng::math::vec3<f64>;

struct kepler final {
    f64 a{};
    f64 e{};
    f64 i{};
    f64 Omega{};
    f64 omega{};
    f64 M0{};
};

struct body final {
    f64 mu{};
    f64 mass{};
    v3 pos{};
    v3 vel{};
};

struct orbit_state final {
    v3 r{};
    v3 v{};
};

constexpr f64 pi() noexcept { return 3.141592653589793238462643383279502884; }

inline f64 wrap2pi(f64 x) noexcept {
    auto t = std::fmod(x, 2.0*pi());
    return (t < 0) ? (t + 2.0*pi()) : t;
}

inline f64 solve_kepler(f64 M, f64 e) noexcept {
    M = wrap2pi(M);
    f64 E = (e < 0.8) ? M : pi();
    for (int it = 0; it < 16; ++it) {
        auto s = std::sin(E);
        auto c = std::cos(E);
        auto f = E - e*s - M;
        auto fp = 1.0 - e*c;
        auto d = f / fp;
        E -= d;
        if (std::abs(d) < 1e-14) break;
    }
    return E;
}

inline std::pair<v3,v3> kepler_to_state(const kepler& k, f64 mu, f64 t) noexcept {
    auto M = k.M0 + std::sqrt(mu/(k.a*k.a*k.a)) * t;
    auto E = solve_kepler(M, k.e);
    auto cE = std::cos(E);
    auto sE = std::sin(E);
    auto n = std::sqrt(mu/(k.a*k.a*k.a));
    auto r = k.a*(1.0 - k.e*cE);
    auto x = k.a*(cE - k.e);
    auto y = k.a*(std::sqrt(1.0-k.e*k.e)*sE);
    auto vx = -k.a*n*sE / (1.0 - k.e*cE);
    auto vy =  k.a*n*std::sqrt(1.0-k.e*k.e)*cE / (1.0 - k.e*cE);

    auto cO = std::cos(k.Omega), sO = std::sin(k.Omega);
    auto co = std::cos(k.omega), so = std::sin(k.omega);
    auto ci = std::cos(k.i), si = std::sin(k.i);

    auto R11 = cO*co - sO*so*ci;
    auto R12 = -cO*so - sO*co*ci;
    auto R21 = sO*co + cO*so*ci;
    auto R22 = -sO*so + cO*co*ci;
    auto R31 = so*si;
    auto R32 = co*si;

    v3 rr{R11*x + R12*y, R21*x + R22*y, R31*x + R32*y};
    v3 vv{R11*vx + R12*vy, R21*vx + R22*vy, R31*vx + R32*vy};
    return {rr, vv};
}

struct system final {
    std::vector<body> bodies;
    f64 G{6.67430e-11};

    explicit system(std::vector<body> b = {}) : bodies(std::move(b)) {}

    std::vector<v3> accel() const {
        std::vector<v3> a(bodies.size());
        for (std::size_t i = 0; i < bodies.size(); ++i) {
            v3 ai{};
            for (std::size_t j = 0; j < bodies.size(); ++j) if (i != j) {
                auto dr = bodies[j].pos - bodies[i].pos;
                auto r2 = dr.norm2() + 1e-12;
                auto invr = 1.0/std::sqrt(r2);
                auto invr3 = invr*invr*invr;
                ai += dr * (G * bodies[j].mass * invr3);
            }
            a[i] = ai;
        }
        return a;
    }

    void step_leapfrog(f64 dt) {
        auto a0 = accel();
        for (std::size_t i = 0; i < bodies.size(); ++i) bodies[i].vel += a0[i] * (0.5*dt);
        for (auto& b : bodies) b.pos += b.vel * dt;
        auto a1 = accel();
        for (std::size_t i = 0; i < bodies.size(); ++i) bodies[i].vel += a1[i] * (0.5*dt);
    }
};

inline kepler random_kepler(eng::math::splitmix64& rng, f64 a0, f64 a1) {
    auto a = rng.uniform01<f64>()*(a1-a0)+a0;
    auto e = std::pow(rng.uniform01<f64>(), 2.2) * 0.25;
    auto i = std::pow(rng.uniform01<f64>(), 1.7) * (10.0*pi()/180.0);
    auto Omega = rng.uniform01<f64>()*(2.0*pi());
    auto omega = rng.uniform01<f64>()*(2.0*pi());
    auto M0 = rng.uniform01<f64>()*(2.0*pi());
    return kepler{a,e,i,Omega,omega,M0};
}

inline std::vector<orbit_state> synthesize_orbits(std::uint64_t seed, std::size_t n, f64 mu_central) {
    eng::math::splitmix64 rng(seed);
    std::vector<orbit_state> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto k = random_kepler(rng, 0.4e11, 6.0e11);
        auto [r,v] = kepler_to_state(k, mu_central, 0.0);
        out.push_back(orbit_state{r,v});
    }
    return out;
}

}
