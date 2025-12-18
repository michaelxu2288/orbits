#include <iostream>
#include <vector>
#include <chrono>
#include "engine/engine.hpp"

using eng::f64;
using eng::u64;

struct Position { eng::math::vec3<f64> p; };
struct Velocity { eng::math::vec3<f64> v; };
struct Mass { f64 m; };

static std::vector<eng::math::vec3<double>> sample_orbit(const eng::sim::kepler& k, double mu, std::size_t steps) {
    std::vector<eng::math::vec3<double>> pts;
    pts.reserve(steps);
    for (std::size_t i = 0; i < steps; ++i) {
        auto t = (static_cast<double>(i) / static_cast<double>(steps-1)) * (365.25*24.0*3600.0);
        auto [r,v] = eng::sim::kepler_to_state(k, mu, t);
        pts.push_back(r);
    }
    return pts;
}

int main() {
    constexpr double mu_sun = 1.32712440018e20;
    auto orbits = eng::sim::synthesize_orbits(0xA5A5A5A5u, 8, mu_sun);

    eng::core::scheduler sched(4);
    std::vector<std::vector<eng::math::vec3<double>>> tracks(orbits.size());

    for (std::size_t i = 0; i < orbits.size(); ++i) {
        sched.submit([&, i] {
            eng::math::splitmix64 rng(0xC0FFEEu + static_cast<u64>(i)*1315423911ull);
            auto k = eng::sim::random_kepler(rng, 0.6e11, 4.5e11);
            tracks[i] = sample_orbit(k, mu_sun, 512);
        });
    }
    sched.drain();

    eng::gfx::renderer r;
    for (auto& t : tracks) {
        auto m = eng::gfx::orbit_ribbon(t, 5000000.0f);
        r.submit(eng::gfx::draw_packet{std::move(m), 0x1234});
    }

    std::size_t vtx = 0, idx = 0;
    for (auto& p : r.flush()) { vtx += p.m.vtx.size(); idx += p.m.idx.size(); }
    std::cout << "packets=" << r.flush().size() << " vtx=" << vtx << " idx=" << idx << "\n";

    eng::ecs::world<Position, Velocity, Mass> w;
    auto e = w.create();
    w.emplace<Position>(e, Position{eng::math::vec3<f64>{1,2,3}});
    w.emplace<Velocity>(e, Velocity{eng::math::vec3<f64>{0.1,0.2,0.3}});
    w.emplace<Mass>(e, Mass{5.0});

    w.query<Position, Velocity, Mass>([](auto ent, auto& p, auto& v, auto& m){
        std::cout << "entity=" << ent << " p=" << p.p[0] << "," << p.p[1] << "," << p.p[2] << " m=" << m.m << "\n";
        p.p += v.v * 10.0;
    });

    eng::io::bytes b;
    b.put<std::uint64_t>(0xDEADBEEFCAFEBABEull);
    b.put_string("orbit-engine");
    b.put<double>(tracks[0].front()[0]);
    eng::io::reader rd(b.v);
    auto a = rd.get<std::uint64_t>();
    auto s = rd.get_string();
    auto x = rd.get<double>();
    std::cout << std::hex << a << std::dec << " " << s << " " << x << "\n";

    return 0;
}
