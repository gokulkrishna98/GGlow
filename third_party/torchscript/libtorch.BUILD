# Note down in documentation:
# We cannot link libtorch_python.so, because we have not linked python libraries
# we will get undefined refernece to libraries to python types and functions.

# After Adding python libraries to environment we can add it.

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
        # "lib/libtorch_python.so"
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
        # "-ltorch_python"
    ],
    hdrs = glob(["include/**/*.h"]),
    includes = [
        "include",
        "include/torch/csrc/api/include"
    ],

    copts = ["-D_GLIBCXX_USE_CXX11_ABI=0"],
    visibility = ["//visibility:public"]
)