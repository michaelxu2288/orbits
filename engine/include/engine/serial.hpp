#pragma once
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace eng::io {

struct bytes final {
    std::vector<std::byte> v;

    void reserve(std::size_t n) { v.reserve(n); }

    template<class T>
    requires std::is_trivially_copyable_v<T>
    void put(const T& x) {
        auto p = reinterpret_cast<const std::byte*>(&x);
        v.insert(v.end(), p, p + sizeof(T));
    }

    void put_span(std::span<const std::byte> s) { v.insert(v.end(), s.begin(), s.end()); }

    void put_string(std::string_view s) {
        std::uint32_t n = static_cast<std::uint32_t>(s.size());
        put(n);
        auto p = reinterpret_cast<const std::byte*>(s.data());
        v.insert(v.end(), p, p + s.size());
    }
};

struct reader final {
    std::span<const std::byte> s;
    std::size_t i{};

    explicit reader(std::span<const std::byte> in) : s(in), i(0) {}

    template<class T>
    requires std::is_trivially_copyable_v<T>
    T get() {
        if (i + sizeof(T) > s.size()) throw std::runtime_error("eof");
        T x{};
        std::memcpy(&x, s.data() + i, sizeof(T));
        i += sizeof(T);
        return x;
    }

    std::string get_string() {
        auto n = get<std::uint32_t>();
        if (i + n > s.size()) throw std::runtime_error("eof");
        std::string out(reinterpret_cast<const char*>(s.data() + i), reinterpret_cast<const char*>(s.data() + i + n));
        i += n;
        return out;
    }
};

}
