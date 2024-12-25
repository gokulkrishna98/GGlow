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
        "CanonicalizePatterns.td",
        "GGlowOpsInterface.td"
    ],
    deps = [
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:CastInterfacesTdFiles",
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
        ":td_files",
    ]
)

gentbl_cc_library(
    name = "ops_interface_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "GGlowOpsInterface.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "GGlowOpsInterface.cpp.inc",
        )
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "GGlowOpsInterface.td",
    deps = [
        ":dialect_inc_gen",
        ":td_files",
    ]
)

cc_library(
    name = "GGlowDialect",
    srcs = [
        "GGlowDialect.cc",
        "GGlowPass.cc",
        "CanonicalizePatterns.cc",
        "LoweringToAffine.cpp",
        "LoweringToLLVM.cpp"
    ],
    hdrs = [
        "GGlowDialect.h",
        "GGlowPass.h"
    ],
    deps = [
        ":dialect_inc_gen",
        ":ops_inc_gen",
        ":canonicalize_patterns_inc_gen",
        ":ops_interface_inc_gen",

        # mlir dependency
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:CastInterfaces",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:ExecutionEngine",
        # llvm dependency
        "@llvm-project//llvm:Support",
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