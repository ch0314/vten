/**
 * vten_xrt — pybind11 bindings for XRT C++ API.
 *
 * Exposes xrt::device, xrt::xclbin, xrt::bo, xrt::ip to Python.
 * Replaces vendor pyxrt which lacks xrt::ip and is Python 3.8 only.
 *
 * API surface matches what vten/backend/xrt.py and
 * vten/runtime/interpreter.py expect from pyxrt.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include "experimental/xrt_xclbin.h"
#include "experimental/xrt_ip.h"
#include "xrt/xrt_kernel.h"

namespace py = pybind11;


PYBIND11_MODULE(vten_xrt, m) {
    m.doc() = "vTen XRT bindings — xrt::device, xrt::xclbin, xrt::bo, xrt::ip";

    // ── xclBOSyncDirection ──
    // Exposed as a submodule to match pyxrt API:
    //   vten_xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    auto sync_mod = m.def_submodule("xclBOSyncDirection",
        "Buffer synchronization direction constants");
    sync_mod.attr("XCL_BO_SYNC_BO_TO_DEVICE") =
        py::int_(static_cast<int>(XCL_BO_SYNC_BO_TO_DEVICE));
    sync_mod.attr("XCL_BO_SYNC_BO_FROM_DEVICE") =
        py::int_(static_cast<int>(XCL_BO_SYNC_BO_FROM_DEVICE));

    // ── xrt::uuid ──
    py::class_<xrt::uuid>(m, "uuid")
        .def(py::init<>())
        .def("to_string", &xrt::uuid::to_string)
        .def("__repr__", [](const xrt::uuid& u) {
            return "<xrt.uuid '" + u.to_string() + "'>";
        })
        .def("__bool__", [](const xrt::uuid& u) {
            return static_cast<bool>(u);
        });

    // ── xrt::xclbin::ip ──
    py::class_<xrt::xclbin::ip>(m, "xclbin_ip")
        .def("get_name", &xrt::xclbin::ip::get_name,
            "Get IP/CU name (e.g., 'weight_loader:weight_loader_1')");

    // ── xrt::xclbin::kernel ──
    py::class_<xrt::xclbin::kernel>(m, "xclbin_kernel")
        .def("get_name", &xrt::xclbin::kernel::get_name,
            "Get kernel name")
        .def("get_cus",
            py::overload_cast<>(&xrt::xclbin::kernel::get_cus, py::const_),
            "Get list of compute units (xclbin_ip objects)");

    // ── xrt::xclbin ──
    py::class_<xrt::xclbin>(m, "xclbin")
        .def(py::init<const std::string&>(), py::arg("fnm"),
            "Load xclbin from file path")
        .def("get_uuid", &xrt::xclbin::get_uuid,
            "Get UUID of the xclbin")
        .def("get_kernels", &xrt::xclbin::get_kernels,
            "Get list of kernels in the xclbin");

    // ── xrt::device ──
    py::class_<xrt::device>(m, "device")
        .def(py::init<unsigned int>(), py::arg("index"),
            "Open device by index")
        .def("load_xclbin",
            py::overload_cast<const xrt::xclbin&>(
                &xrt::device::load_xclbin),
            py::arg("xclbin"),
            "Load xclbin onto device, returns UUID")
        .def("load_xclbin",
            py::overload_cast<const std::string&>(
                &xrt::device::load_xclbin),
            py::arg("fnm"),
            "Load xclbin from file path onto device, returns UUID");

    // ── xrt::bo ──
    auto bo_cls = py::class_<xrt::bo>(m, "bo")
        .def(py::init<const xrt::device&, size_t, xrt::bo::flags,
                       unsigned int>(),
            py::arg("device"), py::arg("size"),
            py::arg("flags"), py::arg("group"),
            "Allocate buffer object on specified memory bank")
        .def("size", &xrt::bo::size,
            "Buffer size in bytes")
        .def("address", &xrt::bo::address,
            "Device physical address of the buffer")
        .def("sync",
            [](xrt::bo& bo, int dir) {
                bo.sync(static_cast<xclBOSyncDirection>(dir));
            },
            py::arg("dir"),
            "Synchronize entire buffer (to/from device)")
        .def("sync",
            [](xrt::bo& bo, int dir, size_t size, size_t offset) {
                bo.sync(static_cast<xclBOSyncDirection>(dir), size, offset);
            },
            py::arg("dir"), py::arg("size"), py::arg("offset"),
            "Synchronize partial buffer (to/from device)")
        // write: Python bytes → device buffer
        .def("write", [](xrt::bo& bo, py::bytes data) {
            std::string s = data;
            bo.write(s.data(), s.size(), 0);
        }, py::arg("data"),
            "Write bytes to buffer (from offset 0)")
        .def("write", [](xrt::bo& bo, py::bytes data, size_t seek) {
            std::string s = data;
            bo.write(s.data(), s.size(), seek);
        }, py::arg("data"), py::arg("seek"),
            "Write bytes to buffer at specified offset")
        // read: device buffer → Python bytes
        .def("read", [](xrt::bo& bo, size_t size) -> py::bytes {
            std::vector<char> buf(size);
            bo.read(buf.data(), size, 0);
            return py::bytes(buf.data(), size);
        }, py::arg("size"),
            "Read bytes from buffer (from offset 0)")
        .def("read", [](xrt::bo& bo, size_t size,
                         size_t skip) -> py::bytes {
            std::vector<char> buf(size);
            bo.read(buf.data(), size, skip);
            return py::bytes(buf.data(), size);
        }, py::arg("size"), py::arg("skip"),
            "Read bytes from buffer at specified offset")
        // map: establish host memory mapping (matches C++ bo.map<T*>())
        // Returns bytes from the mapped host buffer.
        .def("map_read", [](xrt::bo& bo, size_t size) -> py::bytes {
            auto ptr = bo.map<char*>();
            if (!ptr) {
                throw std::runtime_error("bo.map() returned null");
            }
            return py::bytes(ptr, size);
        }, py::arg("size"),
            "Map host buffer and read bytes from it")
        // map_init: establish mapping, useful as side-effect for hw_emu
        .def("map_init", [](xrt::bo& bo) {
            auto ptr = bo.map<char*>();
            if (!ptr) {
                throw std::runtime_error("bo.map() returned null");
            }
        }, "Establish host memory mapping (hw_emu compatibility)")
        .def("__bool__", [](const xrt::bo& bo) {
            return static_cast<bool>(bo);
        });

    // bo.flags nested enum
    py::enum_<xrt::bo::flags>(bo_cls, "flags")
        .value("normal",      xrt::bo::flags::normal)
        .value("cacheable",   xrt::bo::flags::cacheable)
        .value("device_only", xrt::bo::flags::device_only)
        .value("host_only",   xrt::bo::flags::host_only)
        .value("p2p",         xrt::bo::flags::p2p)
        .value("svm",         xrt::bo::flags::svm);

    // ── xrt::ip ──
    py::class_<xrt::ip>(m, "ip")
        .def(py::init<const xrt::device&, const xrt::uuid&,
                       const std::string&>(),
            py::arg("device"), py::arg("xclbin_id"), py::arg("name"),
            "Open IP with exclusive access for register read/write")
        .def("write_register", &xrt::ip::write_register,
            py::arg("offset"), py::arg("data"),
            "Write 32-bit value to register at offset")
        .def("read_register", &xrt::ip::read_register,
            py::arg("offset"),
            "Read 32-bit value from register at offset");

    // ── kernel access modes (must be registered before xrt::kernel) ──
    py::enum_<xrt::kernel::cu_access_mode>(m, "cu_access_mode")
        .value("exclusive", xrt::kernel::cu_access_mode::exclusive)
        .value("shared", xrt::kernel::cu_access_mode::shared)
        .value("none", xrt::kernel::cu_access_mode::none);

    // ── xrt::kernel ──
    // Used for BO memory group queries and managed execution.
    py::class_<xrt::kernel>(m, "kernel")
        .def(py::init<const xrt::device&, const xrt::uuid&,
                       const std::string&,
                       xrt::kernel::cu_access_mode>(),
            py::arg("device"), py::arg("xclbin_id"), py::arg("name"),
            py::arg("mode") = xrt::kernel::cu_access_mode::shared,
            "Open kernel for argument management")
        .def("group_id", &xrt::kernel::group_id,
            py::arg("argno"),
            "Get memory bank group ID for kernel argument");
}
