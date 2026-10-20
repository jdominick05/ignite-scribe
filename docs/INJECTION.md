# Injection flags

ignite-scribe marks screen text that looks written to instruct an AI agent. The mark goes in a window's `flags`,
on change events and on `wait_for` hits. It is computed locally at capture time, on text that redaction has
already scrubbed. It is metadata, never a filter, and a window without a flag is not thereby safe.

Prompt injection on a screen is mostly **indirect**: the attack sits inside data the agent reads, such as a web
page, an email or a chat message. It is not typed by the user. Public sets are mostly direct, so each set below
is reported separately.

## Tier 1: patterns (`injection.py`, shipped)

- Text is normalized before matching: NFKC, casefold, invisible characters dropped, whitespace collapsed.
- Invisible characters are a finding of their own: tag characters, bidi controls, zero-width runs.
- Kinds: `hidden`, `override` (English, German, French, Spanish), `extract`, `role`, `address`, `markup`,
  `secrecy`, `exfil`.
- Only bounded quantifiers are used, and at most 32k characters a block are scanned.
- Results are cached by a digest of each block. This adds 0.23 ms to a watcher refresh, measured on Desktop 2.
- The patterns were tuned on the train splits only.

## The data (`tools/injection_eval.py`, test splits)

| set | licence | what it measures |
|---|---|---|
| deepset/prompt-injections | Apache-2.0 | direct injections and ordinary prompts, English and German, 60 + 56 |
| Lakera/gandalf_ignore_instructions | MIT | direct attacks only, 112 |
| microsoft/BIPIA, email | MIT | 125 attack instructions planted in its emails (middle and end: 250), and the 50 emails alone |
| seed-100 UI lines (this repo) | n/a | 500 synthetic GDI lines, benign |

## Results

| set | patterns: recall / false positives | off-the-shelf classifier, threshold 0.5 | either |
|---|---|---|---|
| deepset | 26.7% / 0% | 36.7% / 0% | 46.7% / 0% |
| Gandalf | 64.3% / n/a | 100% / n/a | 100% / n/a |
| BIPIA, planted in emails | 12.0% / 0% of emails | 47.6% / **62%** of emails | 53.6% / 62% |
| UI lines | n/a / 0% | n/a / **11.8%** | n/a / 11.8% |

The classifier is protectai/deberta-v3-base-prompt-injection-v2 (Apache-2.0, 184M parameters, ONNX fp32 on the
CPU), at 33.9 ms per 512-token window. Its threshold does not rescue it:

| threshold | deepset recall | BIPIA recall | BIPIA emails flagged | UI lines flagged |
|---:|---:|---:|---:|---:|
| 0.5 | 36.7% | 47.6% | 62% | 11.8% |
| 0.9 | 33.3% | 43.6% | 60% | 9.2% |
| 0.99 | 30.0% | 37.2% | 50% | 6.2% |
| 0.999 | 21.7% | 24.4% | 34% | 3.2% |

It was trained on prompts typed at chatbots. On screen text it flags ordinary emails and UI lines, and a window
holds hundreds of lines, so nearly every window would be flagged. Off-the-shelf classifiers therefore do not
ship here. The module that runs one (`injection_model.py`) stays, for evaluation and for the model below.

## Tier 2: a classifier trained on screen text (planned)

This follows the ScribeRec route: a model trained for this domain, small and int8-friendly, so it runs exactly
on the NPU in the vision worker. A CPU fallback exists only for measurement.
- **Positives:** injections from the permissive sets' train splits, plus generated ones. They are planted in
  screen-like contexts (pages, emails, chat, documents, terminals, UI) at random positions, with the kinds above
  and paraphrases.
- **Negatives:** screen-like text: UI lines, emails, documents, code, terminal output, chat. Also hard negatives:
  text *about* prompt injection, such as security articles, this file, and detector code. Agents read such text
  all day, and it is the obvious false-positive class.
- **The target:** recall on planted BIPIA attacks well above the patterns' 12%, with false positives on benign
  emails and UI lines in the patterns' range (0 in these sets). Judged per set, on held-out data, on the NPU.
- The model and its recipe live in ignite-xdna, like ScribeRec. Training runs on Desktop 1's GPU.
