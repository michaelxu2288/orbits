#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "engine/vec.hpp"

namespace eng::gfx {

using f32 = float;
using v2 = eng::math::vec2<f32>;
using v3 = eng::math::vec3<f32>;
using v4 = eng::math::vec4<f32>;

struct vertex final {
    v3 pos;
    v3 nrm;
    v2 uv;
    v4 rgba;
};

struct mesh final {
    std::vector<vertex> vtx;
    std::vector<std::uint32_t> idx;
};

struct draw_packet final {
    mesh m;
    std::uint64_t material_tag{};
};

class renderer final {
    std::vector<draw_packet> packets;

public:
    void clear() { packets.clear(); }
    void submit(draw_packet p) { packets.emplace_back(std::move(p)); }
    std::span<const draw_packet> flush() const { return packets; }
};

inline mesh orbit_ribbon(std::span<const eng::math::vec3<double>> pts, float width) {
    mesh m;
    if (pts.size() < 2) return m;
    m.vtx.reserve(pts.size()*2);
    m.idx.reserve((pts.size()-1)*6);
    for (std::size_t i = 0; i < pts.size(); ++i) {
        auto p = pts[i];
        auto t = (i+1 < pts.size()) ? (pts[i+1] - pts[i]).normalized() : (pts[i] - pts[i-1]).normalized();
        eng::math::vec3<double> up{0.0, 0.0, 1.0};
        auto b = eng::math::cross(t, up);
        if (b.norm2() < 1e-18) { up = eng::math::vec3<double>{0.0, 1.0, 0.0}; b = eng::math::cross(t, up); }
        b = b.normalized();
        auto o = b * static_cast<double>(width);
        auto p0 = p + o;
        auto p1 = p - o;

        vertex a{
            v3{static_cast<float>(p0[0]), static_cast<float>(p0[1]), static_cast<float>(p0[2])},
            v3{0,0,1},
            v2{0,0},
            v4{1,1,1,1}
        };
        vertex c{
            v3{static_cast<float>(p1[0]), static_cast<float>(p1[1]), static_cast<float>(p1[2])},
            v3{0,0,1},
            v2{1,0},
            v4{1,1,1,1}
        };
        m.vtx.push_back(a);
        m.vtx.push_back(c);
    }
    for (std::uint32_t i = 0; i + 2 < static_cast<std::uint32_t>(pts.size()*2); i += 2) {
        auto i0 = i;
        auto i1 = i+1;
        auto i2 = i+2;
        auto i3 = i+3;
        m.idx.insert(m.idx.end(), {i0,i2,i1, i1,i2,i3});
    }
    return m;
}

}
