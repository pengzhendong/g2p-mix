# Clean CPP test provenance

`test.sent` and `test.lb` are copied unchanged from `data/cpp` at revision
`9fd7524e6a07e94107f681a8ba735f21df79be61` of
<https://github.com/NewZsh/polyphone>.

The upstream README describes this as CPP with non-polyphonic targets removed
and says that 8,925 test sentences remain. The files at the pinned revision
actually contain 8,935 non-empty, unique, correctly marked sentence/label
pairs. The benchmark uses the files as published and reports 8,935 cases.

The upstream repository does not contain a license file, so the recorded SPDX
license value is `NOASSERTION`. The original CPP corpus in `kakaobrain/g2pM`
is published under Apache-2.0.
