"""
warpper bazel cc_xx to modify flags.
"""

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")
load("@yacl//bazel:yacl.bzl", "yacl_cmake_external")

WARNING_FLAGS = [
    "-Wall",
    "-Wextra",
    "-Werror",
    "-Wno-unused-parameter",
]

DEBUG_FLAGS = ["-O0", "-g"]
RELEASE_FLAGS = ["-O2"]
FAST_FLAGS = ["-O1"]

def _spu_copts():
    return select({
        "@spulib//bazel:spu_build_as_release": RELEASE_FLAGS,
        "@spulib//bazel:spu_build_as_debug": DEBUG_FLAGS,
        "@spulib//bazel:spu_build_as_fast": FAST_FLAGS,
        "//conditions:default": FAST_FLAGS,
    }) + WARNING_FLAGS

def spu_cc_binary(
        linkopts = [],
        copts = [],
        **kargs):
    cc_binary(
        linkopts = linkopts,
        copts = copts + _spu_copts(),
        linkstatic = True,
        **kargs
    )

def spu_cc_library(
        linkopts = [],
        copts = [],
        deps = [],
        local_defines = [],
        **kargs):
    cc_library(
        linkopts = linkopts,
        copts = _spu_copts() + copts,
        deps = deps + [
            "@com_github_gabime_spdlog//:spdlog",
        ],
        local_defines = local_defines + [
            "SPU_BUILD",
        ],
        linkstatic = True,
        **kargs
    )

spu_cmake_external = yacl_cmake_external

def _spu_version_file_impl(ctx):
    out = ctx.actions.declare_file(ctx.attr.filename)
    ctx.actions.write(
        output = out,
        content = "__version__ = \"{}\"\n".format(ctx.attr.version),
    )
    return [DefaultInfo(files = depset([out]))]

spu_version_file = rule(
    implementation = _spu_version_file_impl,
    attrs = {
        "version": attr.string(),
        "filename": attr.string(),
    },
)

def spu_cc_test(
        linkopts = [],
        copts = [],
        deps = [],
        local_defines = [],
        **kwargs):
    cc_test(
        # -lm for tcmalloc
        linkopts = linkopts + ["-lm"],
        copts = _spu_copts() + copts,
        deps = deps + [
            "@com_google_googletest//:gtest_main",
            "@yacl//yacl/utils:elapsed_timer",
        ],
        local_defines = local_defines + [
            "SPU_BUILD",
        ],
        linkstatic = True,
        **kwargs
    )
