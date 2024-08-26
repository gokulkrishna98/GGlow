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
        "CanonicalizePatterns.td"
    ],
    deps = [
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
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

gentbl_cc_library(
    name = "ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "GGlowOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "GGlowOps.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "GGlowOps.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
        # ":types_inc_gen",
    ],
)

gentbl_cc_library(
    name = "canonicalize_patterns_inc_gen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "CanonicalizePatterns.inc",
        )
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "CanonicalizePatterns.td",
    deps = [
        ":dialect_inc_gen",
        ":ops_inc_gen",
        "td_files",
    ]
)


cc_library(
    name = "GGlowDialect",
    srcs = [
        "GGlowDialect.cc",
        "CanonicalizePatterns.cc"
    ],
    hdrs = [
        "GGlowDialect.h",
    ],
    deps = [
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":canonicalize_patterns_inc_gen",
        # ":types_inc_gen",
        # mlir dependency
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:AllPassesAndDialects",
        # llvm dependency
        "@llvm-project//llvm:Support"
    ],
)

cc_library(
    name = "GGlowLib",
    srcs = [
        "GGlow.cc"
    ],
    hdrs = [
        "GGlow.h"
    ],
    deps = [
        ":GGlowDialect",
    ],
)