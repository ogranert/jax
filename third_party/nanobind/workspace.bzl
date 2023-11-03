# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loads the nanobind library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "nanobind",
        strip_prefix = "nanobind-1.5.2",
        sha256 = "2574b91ba15d6160cbc819eb72da3c885601b0468e0d9eda83fc14d3be996113",
        urls = tf_mirror_urls("https://github.com/wjakob/nanobind/archive/refs/tags/v1.5.2.tar.gz"),
        build_file = "//third_party/nanobind:BUILD.bazel",
    )
