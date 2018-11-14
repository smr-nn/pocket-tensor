/*
 * pocket-tensor (c) 2018 Gustavo Valiente gustavo.valiente.m@gmail.com
 * Kerasify (c) 2016 Robert W. Rose
 *
 * MIT License, see LICENSE file.
 */

#ifndef HIKE_THREAD_POOL_H
#define HIKE_THREAD_POOL_H

#include <vector>
#include <functional>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace pt
{

class Dispatcher
{

public:
    using Task = std::function<void(void)>;

    Dispatcher();

    explicit Dispatcher(std::size_t threads);

    ~Dispatcher();

    std::size_t threads() const noexcept
    {
        return _threadsCount;
    }

    void add(Task&& task);

    std::size_t pendingTasks() noexcept;

    void join();

protected:
    std::mutex _mutex;
    std::condition_variable _condition;
    std::deque<Task> _tasks;
    std::vector<std::thread> _threads;
    std::size_t _threadsCount;
    bool _exit;

    std::mutex _pendingTasksMutex;
    std::condition_variable _pendingTasksCondition;
    std::size_t _pendingTasks;
};

}

#endif
