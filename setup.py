# Copyright (c) 2023 Zhendong Peng (pzd17@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages


with open("requirements.txt", encoding="utf8") as f:
    requirements = f.readlines()
extras_require = {"g2pw": ["torch", "modelscope", "pypinyin-g2pw"]}

setup(
    name="g2p-mix",
    version=open("VERSION", encoding="utf8").read(),
    author="Zhendong Peng",
    author_email="pzd17@tsinghua.org.cn",
    long_description=open("README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    description="G2P mix",
    url="https://github.com/pengzhendong/g2p-mix",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "g2p_mix = g2p_mix.cli:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
)
