# Copyright (c) 2024, Zhendong Peng (pzd17@tsinghua.org.cn)
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

import click

from g2p_mix import G2pMix


@click.command()
@click.argument("text")
@click.option("--jyut/--no-jyut", default=False)
@click.option("--sandhi/--no-sandhi", default=True)
@click.option("--return-seg/--no-return-seg", default=False)
def main(text, jyut, sandhi, return_seg):
    g2per = G2pMix(jyut)
    res = g2per.g2p(text, sandhi, return_seg)
    print(res)


if __name__ == "__main__":
    main()
