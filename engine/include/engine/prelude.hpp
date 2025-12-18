#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <future>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <queue>
#include <random>
#include <ranges>
#include <shared_mutex>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace eng {
using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using f32 = float;
using f64 = double;

template<class... Ts>
struct overload : Ts... { using Ts::operator()...; };

template<class... Ts>
overload(Ts...) -> overload<Ts...>;

template<class T>
constexpr T rotl(T x, int r) noexcept {
    if constexpr (sizeof(T) == 8) return static_cast<T>(std::rotl(static_cast<u64>(x), r));
    if constexpr (sizeof(T) == 4) return static_cast<T>(std::rotl(static_cast<u32>(x), r));
    return x;
}

}
