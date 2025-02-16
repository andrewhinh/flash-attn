#include <queue>
#include <stdexcept>

class Queue {
public:
    Queue() {}

    // Push an integer into the queue.
    void push(int value) {
        q.push(value);
    }

    // Pop an integer from the queue.
    int pop() {
        if (q.empty())
            throw std::runtime_error("Queue is empty");
        int front = q.front();
        q.pop();
        return front;
    }

    // Check if the queue is empty.
    bool empty() const {
        return q.empty();
    }

private:
    std::queue<int> q;
};