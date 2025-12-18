#pragma once
#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace eng::util {

template<class T, std::size_t N>
class smallvec final {
    alignas(T) unsigned char buf[sizeof(T)*N];
    T* heap{};
    std::size_t sz{};
    std::size_t cap{N};

    constexpr T* local() noexcept { return std::launder(reinterpret_cast<T*>(buf)); }
    constexpr const T* local() const noexcept { return std::launder(reinterpret_cast<const T*>(buf)); }
    constexpr bool on_heap() const noexcept { return heap != nullptr; }
    constexpr T* ptr() noexcept { return on_heap() ? heap : local(); }
    constexpr const T* ptr() const noexcept { return on_heap() ? heap : local(); }

    void grow(std::size_t ncap) {
        T* np = static_cast<T*>(::operator new(sizeof(T)*ncap, std::align_val_t(alignof(T))));
        for (std::size_t i = 0; i < sz; ++i) std::construct_at(np+i, std::move(ptr()[i]));
        clear_storage();
        heap = np;
        cap = ncap;
    }

    void clear_storage() {
        for (std::size_t i = 0; i < sz; ++i) std::destroy_at(ptr()+i);
        if (on_heap()) ::operator delete(heap, std::align_val_t(alignof(T)));
        heap = nullptr;
        cap = N;
    }

public:
    smallvec() = default;

    smallvec(std::initializer_list<T> il) {
        for (auto& x : il) push_back(x);
    }

    smallvec(const smallvec& o) {
        for (std::size_t i = 0; i < o.sz; ++i) push_back(o.ptr()[i]);
    }

    smallvec& operator=(const smallvec& o) {
        if (this == &o) return *this;
        clear();
        for (std::size_t i = 0; i < o.sz; ++i) push_back(o.ptr()[i]);
        return *this;
    }

    smallvec(smallvec&& o) noexcept {
        if (o.on_heap()) {
            heap = std::exchange(o.heap, nullptr);
            sz = std::exchange(o.sz, 0);
            cap = std::exchange(o.cap, N);
        } else {
            for (std::size_t i = 0; i < o.sz; ++i) push_back(std::move(o.ptr()[i]));
            o.clear();
        }
    }

    smallvec& operator=(smallvec&& o) noexcept {
        if (this == &o) return *this;
        clear_storage();
        sz = 0;
        if (o.on_heap()) {
            heap = std::exchange(o.heap, nullptr);
            sz = std::exchange(o.sz, 0);
            cap = std::exchange(o.cap, N);
        } else {
            for (std::size_t i = 0; i < o.sz; ++i) push_back(std::move(o.ptr()[i]));
            o.clear();
        }
        return *this;
    }

    ~smallvec() { clear_storage(); }

    void clear() {
        for (std::size_t i = 0; i < sz; ++i) std::destroy_at(ptr()+i);
        sz = 0;
    }

    std::size_t size() const noexcept { return sz; }
    std::size_t capacity() const noexcept { return cap; }
    bool empty() const noexcept { return sz == 0; }

    T& operator[](std::size_t i) noexcept { return ptr()[i]; }
    const T& operator[](std::size_t i) const noexcept { return ptr()[i]; }

    T* begin() noexcept { return ptr(); }
    T* end() noexcept { return ptr() + sz; }
    const T* begin() const noexcept { return ptr(); }
    const T* end() const noexcept { return ptr() + sz; }

    template<class... A>
    T& emplace_back(A&&... a) {
        if (sz == cap) grow(cap + cap/2 + 1);
        std::construct_at(ptr() + sz, std::forward<A>(a)...);
        return ptr()[sz++];
    }

    void push_back(const T& x) { emplace_back(x); }
    void push_back(T&& x) { emplace_back(std::move(x)); }

    void pop_back() {
        if (!sz) return;
        std::destroy_at(ptr() + (sz-1));
        --sz;
    }
};

}
