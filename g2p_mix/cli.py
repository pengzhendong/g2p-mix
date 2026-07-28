# SPDX-License-Identifier: Apache-2.0

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
@click.option(
    "--fallback-backend",
    type=str,
    default=None,
    help="Compatible Chinese backend used when the primary backend fails.",
)
@click.option(
    "--unknown",
    type=click.Choice(["strict", "preserve"]),
    default="strict",
    show_default=True,
    help="How to handle Chinese characters without a pronunciation.",
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
def main(
    text,
    mode,
    backend,
    fallback_backend,
    unknown,
    tone_sandhi,
    output,
    output_format,
):
    try:
        converter = G2P(
            mode,
            output=output,
            backend=backend,
            fallback_backend=fallback_backend,
            unknown=unknown,
            tone_sandhi=tone_sandhi,
        )
        result = converter(text)
        if output_format == "json":
            payload = {
                "text": result.original_text,
                "normalized_text": result.normalized_text,
                "output": result.output,
                "phones": result.phones,
                "base_phones": result.base_phones,
                "warnings": result.warnings,
                "tokens": [
                    {
                        "text": item.token.text,
                        "language": item.token.language.value,
                        "pos": item.token.pos,
                        "units": [
                            {
                                "text": unit.text,
                                "base_phones": unit.phones,
                                "tone": unit.tone,
                                "alphabet": unit.alphabet.value,
                                "source_alphabet": (
                                    unit.source_alphabet.value if unit.source_alphabet is not None else None
                                ),
                                "source_phones": unit.source_phones,
                                "tone_contour": unit.tone_contour,
                                "stress_marks": unit.stress_marks,
                                "is_unknown": unit.is_unknown,
                            }
                            for unit in item.units
                        ],
                    }
                    for item in result.tokens
                ],
            }
            click.echo(json.dumps(payload, ensure_ascii=False))
            return

        for warning in result.warnings:
            click.echo(f"warning: {warning}", err=True)
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
