workspace(name = "submodule_example")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
SKYLIB_VERSION = "1.3.0"

http_archive(
    name = "bazel_skylib",
    sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/{version}/bazel-skylib-{version}.tar.gz".format(version = SKYLIB_VERSION),
        "https://github.com/bazelbuild/bazel-skylib/releases/download/{version}/bazel-skylib-{version}.tar.gz".format(version = SKYLIB_VERSION),
    ],
)

new_local_repository(
    name = "llvm-raw",
    build_file_content = "# empty",
    path = "third_party/llvm-project",
)
load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")
llvm_configure(name = "llvm-project")


# ADDING TORCHSCRIPT DEPENDENCIES
http_archive(
    name = "libtorch",
    strip_prefix = "libtorch",
    sha256 = "9d16cc0da41e057f20c0be5f26d7418f969e857631cfcb86550ccdecfee8de60",
    urls = ["https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcpu.zip"],
    build_file = "//third_party/torchscript:libtorch.BUILD",
)

# EXTERNAL DEPENDENCIES NEEDED BY LLVM BAZEL
maybe(
    http_archive,
    name = "llvm_zlib",
    build_file = "@llvm-raw//utils/bazel/third_party_build:zlib-ng.BUILD",
    sha256 = "e36bb346c00472a1f9ff2a0a4643e590a254be6379da7cddd9daeb9a7f296731",
    strip_prefix = "zlib-ng-2.0.7",
    urls = [
        "https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.0.7.zip",
    ],
)

maybe(
    http_archive,
    name = "llvm_zstd",
    build_file = "@llvm-raw//utils/bazel/third_party_build:zstd.BUILD",
    sha256 = "7c42d56fac126929a6a85dbc73ff1db2411d04f104fae9bdea51305663a83fd0",
    strip_prefix = "zstd-1.5.2",
    urls = [
        "https://github.com/facebook/zstd/releases/download/v1.5.2/zstd-1.5.2.tar.gz",
    ],
)