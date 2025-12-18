#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace eng::mem {

struct arena final {
    std::byte* base{};
    std::size_t cap{};
    std::size_t head{};

    explicit arena(std::size_t bytes) : base(static_cast<std::byte*>(std::aligned_alloc(64, (bytes + 63) & ~std::size_t(63)))), cap(bytes), head(0) {
        if (!base) throw std::bad_alloc{};
    }

    arena(const arena&) = delete;
    arena& operator=(const arena&) = delete;

    arena(arena&& o) noexcept : base(std::exchange(o.base, nullptr)), cap(std::exchange(o.cap, 0)), head(std::exchange(o.head, 0)) {}
    arena& operator=(arena&& o) noexcept {
        if (this == &o) return *this;
        if (base) std::free(base);
        base = std::exchange(o.base, nullptr);
        cap = std::exchange(o.cap, 0);
        head = std::exchange(o.head, 0);
        return *this;
    }

    ~arena() {
        if (base) std::free(base);
    }

    void reset() noexcept { head = 0; }

    void* alloc(std::size_t bytes, std::size_t align = alignof(std::max_align_t)) {
        auto p = reinterpret_cast<std::uintptr_t>(base) + head;
        auto aligned = (p + (align - 1)) & ~(align - 1);
        auto next = aligned - reinterpret_cast<std::uintptr_t>(base) + bytes;
        if (next > cap) throw std::bad_alloc{};
        head = next;
        return reinterpret_cast<void*>(aligned);
    }

    template<class T, class... A>
    T* make(A&&... a) {
        void* p = alloc(sizeof(T), alignof(T));
        return std::construct_at(static_cast<T*>(p), std::forward<A>(a)...);
    }
};

template<class T>
struct arena_alloc final {
    using value_type = T;
    arena* a{};

    constexpr arena_alloc() = default;
    constexpr explicit arena_alloc(arena& ar) : a(&ar) {}

    template<class U>
    constexpr arena_alloc(const arena_alloc<U>& o) : a(o.a) {}

    [[nodiscard]] T* allocate(std::size_t n) {
        return static_cast<T*>(a->alloc(sizeof(T)*n, alignof(T)));
    }

    void deallocate(T*, std::size_t) noexcept {}

    template<class U>
    constexpr bool operator==(const arena_alloc<U>& o) const noexcept { return a == o.a; }

    template<class U>
    constexpr bool operator!=(const arena_alloc<U>& o) const noexcept { return !(*this == o); }
};

}
