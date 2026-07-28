# SPDX-License-Identifier: MIT

import json

import click

from . import G2P, G2PError
from .renderers import IpaRenderer, NativeRenderer


@click.command()
@click.argument("text")
@click.option(
    "--mode",
    type=click.Choice(["mandarin", "cantonese"]),
    default="mandarin",
    show_default=True,
)
@click.option(
    "--backend",
    type=str,
    default=None,
    help="Chinese backend; defaults to pypinyin or tojyutping for the selected mode.",
)
@click.option("--tone-sandhi/--no-tone-sandhi", default=True)
@click.option(
    "--output",
    type=click.Choice(["native", "ipa"]),
    default="native",
    show_default=True,
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
)
def main(text, mode, backend, tone_sandhi, output, output_format):
    try:
        converter = G2P(
            mode,
            output=output,
            backend=backend,
            tone_sandhi=tone_sandhi,
        )
        result = converter(text)
        if output_format == "json":
            payload = {
                "text": result.original_text,
                "normalized_text": result.normalized_text,
                "output": result.output,
                "phones": result.phones,
                "segments": result.segments,
                "tokens": [
                    {
                        "text": item.token.text,
                        "language": item.token.language.value,
                        "pos": item.token.pos,
                        "units": [
                            {
                                "text": unit.text,
                                "segments": unit.phones,
                                "tone": unit.tone,
                                "stress": unit.stress,
                                "alphabet": unit.alphabet.value,
                                "source_alphabet": (
                                    unit.source_alphabet.value if unit.source_alphabet is not None else None
                                ),
                                "source_phones": unit.source_phones,
                                "tone_contour": unit.tone_contour,
                                "stress_marks": unit.stress_marks,
                            }
                            for unit in item.units
                        ],
                    }
                    for item in result.tokens
                ],
            }
            click.echo(json.dumps(payload, ensure_ascii=False))
            return

        renderer = IpaRenderer() if output == "ipa" else NativeRenderer()
        for item in result.tokens:
            phones = [phone for unit in item.units for phone in renderer.render_unit(unit)]
            if not phones:
                continue
            click.echo(f"{item.token.text}\t{' '.join(phones)}")
    except G2PError as error:
        raise click.ClickException(str(error)) from error


if __name__ == "__main__":
    main()
