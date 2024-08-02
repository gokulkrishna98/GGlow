load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
package(
    default_visibility = ["//visibility:public"],
)

td_library(
    name = "td_files",
    srcs = [
        "GGlowDialect.td",
        "GGlowOps.td",
        "GGlowTypes.td",
    ],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "dialect_inc_gen",
    tbl_outs = [
        (
            ["-gen-dialect-decls"],
            "GGlowDialect.h.inc",
        ),
        (
            ["-gen-dialect-defs"],
            "GGlowDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "GGlowDialect.td",
    deps = [
        ":td_files",
    ],
)

# gentbl_cc_library(
#     name = "types_inc_gen",
#     tbl_outs = [
#         (
#             ["-gen-dialect-decls"],
#             "GGlowTypes.h.inc",
#         ),
#         (
#             ["-gen-dialect-defs"],
#             "GGlowTypes.cpp.inc",
#         ),
#     ],
#     tblgen = "@llvm-project//mlir:mlir-tblgen",
#     td_file = "GGlowTypes.td",
#     deps = [
#         ":td_files",
#     ],
# )

# gentbl_cc_library(
#     name = "ops_inc_gen",
#     tbl_outs = [
#         (
#             ["-gen-op-decls"],
#             "GGlowOps.h.inc",
#         ),
#         (
#             ["-gen-op-defs"],
#             "GGlowOps.cpp.inc",
#         ),
#     ],
#     tblgen = "@llvm-project//mlir:mlir-tblgen",
#     td_file = "GGlowOps.td",
#     deps = [
#         ":dialect_inc_gen",
#         ":td_files",
#         ":types_inc_gen",
#     ],
# )

cc_library(
    name = "GGlow",
    srcs = [
        "GGlow.cpp"
    ],
    hdrs = [
        "GGlow.h",
    ],
    deps = [
        ":dialect_inc_gen",
        # ":ops_inc_gen",
        # ":types_inc_gen",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
    ],
)