from torch.utils.cpp_extension import load_inline

from utils import PARENT_PATH

QUEUE_FILE = PARENT_PATH / "src" / "queue.cpp"

# C++ source code as a raw string.
source = f"""
#include <pybind11/pybind11.h>
namespace py = pybind11;
{QUEUE_FILE.read_text()}
// Bind the Queue class to Python.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    py::class_<Queue>(m, "Queue")
        .def(py::init<>())
        .def("push", &Queue::push, "Push an integer into the queue")
        .def("pop", &Queue::pop, "Pop an integer from the queue")
        .def("empty", &Queue::empty, "Return whether the queue is empty");
}}
"""

# Compile and load the extension.
queue_ext = load_inline(name="queue_extension", cpp_sources=source, functions=None)

# Use the exposed Queue class.
q = queue_ext.Queue()
q.push(10)
q.push(20)
print("Popped:", q.pop())  # Should print: Popped: 10
print("Is queue empty?", q.empty())  # Should print: Is queue empty? False
