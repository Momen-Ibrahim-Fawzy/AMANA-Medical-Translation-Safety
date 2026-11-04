# Contributing

The checker's weakest evidence is that one author wrote most of its tests. These contributions help most.

## Review translations (clinicians and pharmacists)

Open `adjudication/packet.html` in a browser. It shows pairs of Arabic translations for the same instruction, with
the system and checker hidden. Judge each one and save your answers, then send them through a
[clinical review issue](https://github.com/Momen-Ibrahim-Fawzy/AMANA-Medical-Translation-Safety/issues/new/choose).
Please say what your qualification is. You don't need to know anything about the code. Please don't look in the
`results/` folder first: which version is the system's normal output can be found there, which would unblind the packet.
If you can review more seriously, open an issue and a packet will be built for you from a private seed.

## Write test cases without reading the code

New cases are only useful as evidence if their author hasn't seen the checker's rules. Write English instructions
with Arabic translations, some correct and some with a clinical error, and send them through a
[test cases issue](https://github.com/Momen-Ibrahim-Fawzy/AMANA-Medical-Translation-Safety/issues/new/choose). They
will be frozen by hash and scored once before anyone changes the rules.

Do not include patient information. Use invented or public instructions only.

## Share real errors

Real translated instructions with errors are what the project most lacks. If you have some you are allowed to
share, with identifying details removed, please open an issue and describe the source and licence.

## Change the code

- Run `pytest tests` before opening a pull request.
- New rules need a test for what they catch and a test for what they must not flag.
- Never tune on sealed data: `data/hard_cases_holdout.jsonl`, the sealed drug names, the sealed FDA labels or the sealed PEACH leaflets.
- Report numbers from a run you can reproduce, and say which data shaped the change.
