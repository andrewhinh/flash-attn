from pydantic import BaseModel
from torch.utils.cpp_extension import load_inline

from utils import DIST_PATH, PARENT_PATH

MAPQUEUE_FILE = PARENT_PATH / "src" / "mapqueue.cpp"


class Gen(BaseModel):
    id: int
    query: str
    plotly_plot: str = ""  # needs time to generate


class MapQueue:
    def __init__(self):
        # C++ source code as a raw string.
        source = f"""
        #include <pybind11/pybind11.h>
        namespace py = pybind11;
        {MAPQUEUE_FILE.read_text()}
        // Bind the MapQueue class to Python.
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
            py::class_<MapQueue>(m, "MapQueue")
                .def(py::init<>())
                .def("enqueue", &MapQueue::enqueue, "Enqueue an entry with id, query and plotly_plot")
                .def("getLength", &MapQueue::getLength, "Get the length of the queue")
                .def("getElement", &MapQueue::getElement, "Get an element from the queue by index")
                .def("setElement", &MapQueue::setElement, "Set an element in the queue by index");
                
        }}
        """

        # Compile and load the extension.
        mapqueue_ext = load_inline(
            name="mapqueue_extension",
            cpp_sources=source,
            functions=None,
            build_directory=DIST_PATH,
            verbose=True,
        )

        self._mapqueue = mapqueue_ext.MapQueue()

    def append(self, g: Gen):
        self._mapqueue.enqueue(g.id, g.query, g.plotly_plot)

    def __len__(self):
        return self._mapqueue.getLength()

    def __getitem__(self, index):
        id, data = self._mapqueue.getElement(index)
        query = data["query"] if "query" in data else ""
        plotly_plot = data["plotly_plot"] if "plotly_plot" in data else ""
        return Gen(id=id, query=query, plotly_plot=plotly_plot)

    def __setitem__(self, index, g: Gen):
        self._mapqueue.setElement(index, g.id, g.query, g.plotly_plot)


# Test
mq = MapQueue()

# Test append method
mq.append(Gen(id=1, query="query1", plotly_plot="plot1"))
mq.append(Gen(id=2, query="query2", plotly_plot=""))
mq.append(Gen(id=3, query="", plotly_plot="plot3"))

# Test __len__ method
assert len(mq) == 3, "MapQueue length should be 3"

# Test __getitem__ method
element = mq[0]
assert element == Gen(
    id=1, query="query1", plotly_plot="plot1"
), "First element check failed"

element = mq[1]
assert element == Gen(
    id=2, query="query2", plotly_plot=""
), "Second element check failed"

element = mq[2]
assert element == Gen(id=3, query="", plotly_plot="plot3"), "Third element check failed"

# Test out-of-range exception for __getitem__
try:
    mq[3]
    assert False, "Exception should be thrown for out of range index"
except IndexError:
    # Expected exception
    pass

# Test __setitem__ method
mq[1] = Gen(id=4, query="new_query", plotly_plot="new_plot")
element = mq[1]
assert element == Gen(
    id=4, query="new_query", plotly_plot="new_plot"
), "Set item check failed"

# Test out-of-range exception for __setitem__
try:
    mq[3] = Gen(id=5, query="query5", plotly_plot="plot5")
    assert False, "Exception should be thrown for out of range index"
except IndexError:
    # Expected exception
    pass
