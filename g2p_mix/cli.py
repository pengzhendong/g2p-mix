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

import json

import click

from g2p_mix import G2PWBackend, MixedG2P, NativeRenderer


@click.command()
@click.argument("text")
@click.option(
    "--mode",
    type=click.Choice(["mandarin", "cantonese"]),
    default="mandarin",
    show_default=True,
)
@click.option(
    "--mandarin-backend",
    type=click.Choice(["pypinyin", "g2pw"]),
    default="pypinyin",
    show_default=True,
)
@click.option("--tone-sandhi/--no-tone-sandhi", default=True)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
)
def main(text, mode, mandarin_backend, tone_sandhi, output_format):
    if mode == "mandarin":
        backend = G2PWBackend() if mandarin_backend == "g2pw" else None
        converter = MixedG2P.mandarin(
            chinese_backend=backend,
            tone_sandhi=tone_sandhi,
        )
    else:
        converter = MixedG2P.cantonese()

    result = converter(text)
    if output_format == "json":
        payload = {
            "text": result.original_text,
            "normalized_text": result.normalized_text,
            "tokens": [
                {
                    "text": output.token.text,
                    "language": output.token.language.value,
                    "pos": output.token.pos,
                    "units": [
                        {
                            "text": unit.text,
                            "phones": unit.phones,
                            "tone": unit.tone,
                            "stress": unit.stress,
                            "alphabet": unit.alphabet.value,
                        }
                        for unit in output.units
                    ],
                }
                for output in result.tokens
            ],
        }
        click.echo(json.dumps(payload, ensure_ascii=False))
        return

    renderer = NativeRenderer()
    for output in result.tokens:
        phones = [phone for unit in output.units for phone in renderer.render_unit(unit)]
        if not phones:
            continue
        click.echo(f"{output.token.text}\t{' '.join(phones)}")


if __name__ == "__main__":
    main()
