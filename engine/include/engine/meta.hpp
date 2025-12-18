#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

namespace eng::meta {

template<class T>
struct type_tag { using type = T; };

template<std::size_t... Is>
struct index_seq { };

template<std::size_t N, std::size_t... Is>
struct make_index_seq : make_index_seq<N-1, N-1, Is...> { };

template<std::size_t... Is>
struct make_index_seq<0, Is...> { using type = index_seq<Is...>; };

template<class F, class Tuple, std::size_t... Is>
constexpr decltype(auto) apply_i(F&& f, Tuple&& t, index_seq<Is...>) {
    return std::forward<F>(f)(std::get<Is>(std::forward<Tuple>(t))...);
}

template<class F, class Tuple>
constexpr decltype(auto) apply(F&& f, Tuple&& t) {
    constexpr auto N = std::tuple_size_v<std::remove_reference_t<Tuple>>;
    return apply_i(std::forward<F>(f), std::forward<Tuple>(t), typename make_index_seq<N>::type{});
}

template<class... Ts>
struct typelist { };

template<class T, class List>
struct push_front;

template<class T, class... Ts>
struct push_front<T, typelist<Ts...>> { using type = typelist<T, Ts...>; };

template<class List>
struct size;

template<class... Ts>
struct size<typelist<Ts...>> : std::integral_constant<std::size_t, sizeof...(Ts)> { };

template<class List, template<class> class Pred>
struct filter;

template<template<class> class Pred>
struct filter<typelist<>, Pred> { using type = typelist<>; };

template<class T, class... Ts, template<class> class Pred>
struct filter<typelist<T, Ts...>, Pred> {
    using rest = typename filter<typelist<Ts...>, Pred>::type;
    using type = std::conditional_t<Pred<T>::value, typename push_front<T, rest>::type, rest>;
};

template<class... Ts>
constexpr std::uint64_t signature() {
    std::uint64_t h = 0xcbf29ce484222325ull;
    ((h = (h ^ static_cast<std::uint64_t>(alignof(Ts))) * 0x100000001b3ull), ...);
    ((h = (h ^ static_cast<std::uint64_t>(sizeof(Ts))) * 0x100000001b3ull), ...);
    return h;
}

}
