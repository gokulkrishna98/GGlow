# Note down in documentation:
# added -lpython3.10 (dont know how this linking options worked xd)

cc_library(
    name = "libtorch",
    srcs = [
        "lib/libtorch.so",
        "lib/libc10.so",
        "lib/libbackend_with_compiler.so",
        "lib/libtorch_cpu.so",
        "lib/libnnapi_backend.so",
        "lib/libtorch_global_deps.so",
        "lib/libtorchbind_test.so",
        "lib/libjitbackend_test.so",
        "lib/libshm.so",
        "lib/libtorch_python.so",
        "lib/libgomp-98b21ff3.so.1"
    ],
    linkopts = [
        "-ltorch",
        "-lc10",
        "-lbackend_with_compiler",
        "-ltorch_cpu",
        "-lnnapi_backend",
        "-ltorch_global_deps",
        "-ltorchbind_test",
        "-ljitbackend_test",
        "-lshm",
        "-ltorch_python",
        "-lpython3.10",
        "-lgomp"
    ],
    hdrs = glob(["include/**/*.h"]),
    includes = [
        "include",
        "include/torch/csrc/api/include"
    ],
    copts = ["-D_GLIBCXX_USE_CXX11_ABI=1"],
    visibility = ["//visibility:public"],
)

