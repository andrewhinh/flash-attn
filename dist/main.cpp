#include <torch/extension.h>

        #include <pybind11/pybind11.h>
        namespace py = pybind11;
        #include <queue>
#include <map>
#include <string>
#include <stdexcept>
#include <vector>
#include <cassert>

class MapQueue {
public:
    void enqueue(int id, const std::string& query, const std::string& plotly_plot) {
        std::map<std::string, std::string> mapEntry;
        if (!query.empty()) {
            mapEntry["query"] = query;
        }
        if (!plotly_plot.empty()) {
            mapEntry["plotly_plot"] = plotly_plot;
        }
        queue_.push({id, mapEntry});
    }

    size_t getLength() const {
        return queue_.size();
    }

    std::pair<int, std::map<std::string, std::string>> getElement(size_t index) const {
        if (index >= queue_.size()) {
            throw std::out_of_range("Index is out of range");
        }
        auto tempQueue = queue_;
        for (size_t i = 0; i < index; ++i) {
            tempQueue.pop();
        }
        return tempQueue.front();
    }

    void setElement(size_t index, int id, const std::string& query, const std::string& plotly_plot) {
        if (index >= queue_.size()) {
            throw std::out_of_range("Index is out of range");
        }
        auto tempQueue = std::queue<std::pair<int, std::map<std::string, std::string>>>();
        for (size_t i = 0; i < index; ++i) {
            tempQueue.push(queue_.front());
            queue_.pop();
        }
        std::map<std::string, std::string> mapEntry;
        if (!query.empty()) {
            mapEntry["query"] = query;
        }
        if (!plotly_plot.empty()) {
            mapEntry["plotly_plot"] = plotly_plot;
        }
        queue_.pop();
        tempQueue.push({id, mapEntry});

        while (!queue_.empty()) {
            tempQueue.push(queue_.front());
            queue_.pop();
        }
        queue_ = tempQueue;
    }

    

private:
    std::queue<std::pair<int, std::map<std::string, std::string>>> queue_;
};

// Test Code
int main() {
    MapQueue q;
    
    // Test enqueue method
    q.enqueue(1, "query1", "plot1");
    q.enqueue(2, "query2", "");
    q.enqueue(3, "", "plot3");

    // Test getLength method
    assert(q.getLength() == 3);

    // Test getElement method
    auto element = q.getElement(0);
    assert(element.first == 1 && element.second["query"] == "query1" && element.second["plotly_plot"] == "plot1");

    element = q.getElement(1);
    assert(element.first == 2 && element.second["query"] == "query2" && element.second["plotly_plot"].empty());

    element = q.getElement(2);
    assert(element.first == 3 && element.second["query"].empty() && element.second["plotly_plot"] == "plot3");

    // Test out-of-range exception for getElement
    try {
        q.getElement(3);
        assert(false);
    } catch (const std::out_of_range&) {
        // Expected exception
    }

    // Test setElement method
    q.setElement(1, 4, "new_query", "new_plot");
    element = q.getElement(1);
    assert(element.first == 4 && element.second["query"] == "new_query" && element.second["plotly_plot"] == "new_plot");

    // Test out-of-range exception for setElement
    try {
        q.setElement(3, 5, "query5", "plot5");
        assert(false);
    } catch (const std::out_of_range&) {
        // Expected exception
    }

    return 0;
}

        // Bind the MapQueue class to Python.
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            py::class_<MapQueue>(m, "MapQueue")
                .def(py::init<>())
                .def("enqueue", &MapQueue::enqueue, "Enqueue an entry with id, query and plotly_plot")
                .def("getLength", &MapQueue::getLength, "Get the length of the queue")
                .def("getElement", &MapQueue::getElement, "Get an element from the queue by index")
                .def("setElement", &MapQueue::setElement, "Set an element in the queue by index");
                
        }
        