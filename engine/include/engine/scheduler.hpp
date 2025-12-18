#pragma once
#include <atomic>
#include <barrier>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace eng::core {

class scheduler final {
    struct queue {
        std::mutex m;
        std::deque<std::function<void()>> q;
        void push(std::function<void()> f) {
            std::lock_guard lg(m);
            q.emplace_back(std::move(f));
        }
        std::optional<std::function<void()>> pop() {
            std::lock_guard lg(m);
            if (q.empty()) return std::nullopt;
            auto f = std::move(q.front());
            q.pop_front();
            return f;
        }
        std::optional<std::function<void()>> steal_back() {
            std::lock_guard lg(m);
            if (q.empty()) return std::nullopt;
            auto f = std::move(q.back());
            q.pop_back();
            return f;
        }
    };

    std::vector<std::thread> workers;
    std::vector<std::unique_ptr<queue>> qs;
    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> rr{0};
    std::condition_variable cv;
    std::mutex cv_m;
    std::atomic<std::uint64_t> in_flight{0};

    static std::size_t tid_hash() noexcept {
        auto id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        return static_cast<std::size_t>(id);
    }

    std::optional<std::function<void()>> take(std::size_t i) {
        if (auto f = qs[i]->pop()) return f;
        for (std::size_t k = 1; k < qs.size(); ++k) {
            auto j = (i + k) % qs.size();
            if (auto f = qs[j]->steal_back()) return f;
        }
        return std::nullopt;
    }

    void loop(std::size_t i) {
        for (;;) {
            if (stop.load(std::memory_order_relaxed)) break;
            if (auto f = take(i)) {
                (*f)();
                in_flight.fetch_sub(1, std::memory_order_release);
            } else {
                std::unique_lock lk(cv_m);
                cv.wait_for(lk, std::chrono::microseconds(250), [&]{ return stop.load(std::memory_order_relaxed) || in_flight.load(std::memory_order_acquire) != 0; });
            }
        }
    }

public:
    explicit scheduler(std::size_t n = std::thread::hardware_concurrency()) {
        n = n ? n : 1;
        qs.reserve(n);
        for (std::size_t i = 0; i < n; ++i) qs.emplace_back(std::make_unique<queue>());
        workers.reserve(n);
        for (std::size_t i = 0; i < n; ++i) workers.emplace_back([this, i]{ loop(i); });
    }

    scheduler(const scheduler&) = delete;
    scheduler& operator=(const scheduler&) = delete;

    ~scheduler() {
        stop.store(true, std::memory_order_relaxed);
        cv.notify_all();
        for (auto& t : workers) if (t.joinable()) t.join();
    }

    template<class F>
    void submit(F&& f) {
        in_flight.fetch_add(1, std::memory_order_release);
        auto i = rr.fetch_add(1, std::memory_order_relaxed) % qs.size();
        qs[i]->push(std::function<void()>(std::forward<F>(f)));
        cv.notify_one();
    }

    void drain() {
        while (in_flight.load(std::memory_order_acquire) != 0) {
            std::this_thread::yield();
        }
    }
};

}
