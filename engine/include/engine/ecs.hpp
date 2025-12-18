#pragma once
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "engine/hash.hpp"

namespace eng::ecs {

using entity = std::uint32_t;

struct entity_pool final {
    std::vector<entity> free;
    entity next{1};

    entity make() {
        if (!free.empty()) {
            auto e = free.back();
            free.pop_back();
            return e;
        }
        return next++;
    }

    void destroy(entity e) { free.push_back(e); }
};

template<class C>
struct store final {
    std::vector<entity> owners;
    std::vector<C> data;

    template<class... A>
    C& emplace(entity e, A&&... a) {
        owners.push_back(e);
        data.emplace_back(std::forward<A>(a)...);
        return data.back();
    }

    C* get(entity e) {
        for (std::size_t i = 0; i < owners.size(); ++i) if (owners[i] == e) return &data[i];
        return nullptr;
    }

    const C* get(entity e) const {
        for (std::size_t i = 0; i < owners.size(); ++i) if (owners[i] == e) return &data[i];
        return nullptr;
    }

    void erase(entity e) {
        for (std::size_t i = 0; i < owners.size(); ++i) if (owners[i] == e) {
            owners[i] = owners.back();
            owners.pop_back();
            data[i] = std::move(data.back());
            data.pop_back();
            return;
        }
    }

    template<class F>
    void each(F&& f) {
        for (std::size_t i = 0; i < owners.size(); ++i) f(owners[i], data[i]);
    }
};

template<class... Cs>
class world final {
    entity_pool pool;
    std::tuple<store<Cs>...> stores;

    template<class C>
    store<C>& S() { return std::get<store<C>>(stores); }

    template<class C>
    const store<C>& S() const { return std::get<store<C>>(stores); }

public:
    entity create() { return pool.make(); }
    void destroy(entity e) { (S<Cs>().erase(e), ...); pool.destroy(e); }

    template<class C, class... A>
    C& emplace(entity e, A&&... a) { return S<C>().emplace(e, std::forward<A>(a)...); }

    template<class C>
    C* get(entity e) { return S<C>().get(e); }

    template<class C>
    const C* get(entity e) const { return S<C>().get(e); }

    template<class... Q, class F>
    void query(F&& f) {
        using first_t = std::tuple_element_t<0, std::tuple<Q...>>;
        auto& st = S<first_t>();
        for (std::size_t i = 0; i < st.owners.size(); ++i) {
            entity e = st.owners[i];
            auto* c0 = &st.data[i];
            if (((get<Q>(e) != nullptr) && ...)) {
                f(e, *get<Q>(e)...);
            }
        }
    }
};

}
