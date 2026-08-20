# NLIR

NLIR (Natural Language Intermediate Representation) is a research PoC for a security-focused IR for natural language. The premise of NLIR is to treat LLM input like a non-deterministic programming language, which enables one to see parsing and detection of malicious prompts as a compiler problem.

For more info, see its introductory blog post [Hunting Prompt Malware with NLIR](https://eversinc33.com/hunting-prompt-malware-with-nlir/).

*Disclaimer: This is a research-grade PoC with 100% agent-written code*

The project finds prompt injection and concealed instructions. It safely unpacks Base64, hexadecimal, and URL-encoded text, as well as other custom encodings using an optional reasoning model. NLIR then lifts the source and each unpacked child into the same IR.

NLIR rules are like YARA rules for prompt meaning - they can be used to detect and hunt for semantic behavior.

## How it works

The lifting can be summarized as such:

![The eight-step NLIR lifting pipeline, from loading a file through scanning, static and reasoning-model decoding, semantic lifting, validation, and rule evaluation](docs/diagram.png)

Afterwards rules can be run on the IR. 

The full IR specification — entity types, opcodes, relation types, modality, evidence, and the exact model prompts — is in [`docs/IR.md`](docs/IR.md).

## Install

NLIR requires Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q
```

## Scan a file

```bash
uv run nlir scan file path/to/prompt.md
```

The command prints JSON with source annotations. It does not call a model.

## Use the library API

The library API is the primary interface. The CLI uses the same components.

Create a live lifting session from an explicit non-secret TOML file, lift a file, and run one rule on the result:

```python
from nlir import NLIR

with NLIR.from_live_config("live.toml") as nlir:
    rule = nlir.read_rule("rules/package-install.yaml")
    lifted = nlir.lift_file("path/to/SKILLS.md")
    report = nlir.run_rule(rule, lifted)
```

`lift_file` returns a `LiftedIR` object and nothing else. NLIR never writes IR anywhere: keeping it is your responsibility.

```python
from pathlib import Path

from nlir import LiftedIR

Path("lifted.json").write_text(lifted.model_dump_json(), encoding="utf-8")
restored = LiftedIR.model_validate_json(Path("lifted.json").read_text(encoding="utf-8"))
report = nlir.run_rule(rule, restored)
```

`LiftedIR` is a Pydantic model. Put it in a file, a database, or a queue as you see fit.

`run_rule` also accepts several lift results at once, which is how you hunt everything you have kept:

```python
report = nlir.run_rule(rule, [first_lifted, second_lifted])
```

Use `read_rule_dir` to load all YAML rules from one directory.

## Lift with a live model

Set `NLIR_LIVE_API_KEY` in your shell. Do not put an API key in the TOML file.

Create a local `live.toml` file:

```toml
base_url = "https://api.openai.com/v1"
model = "gpt-4o"
unpack_model = "gpt-4o"
```

`unpack_model` is optional. When you set it, NLIR sends each source artifact to a separate reasoning-unpack request before lifting. That request creates only untrusted virtual children. It does not run commands or use tools.

Lift a file, print its accepted IR, and test one rule:

```bash
uv run nlir lift live path/to/SKILLS.md \
  --config live.toml \
  --show \
  --test-rule rules/package-install.yaml
```

The `--show` option prints the accepted canonical IR for this command. The command writes nothing to disk.

## Open the browser inspector

Start the local browser app with the same live configuration:

```bash
uv run nlir web --config live.toml
```

Open `http://127.0.0.1:5000` in a browser.

![Web preview](./docs/openclaw.png)

The left panel takes a prompt and lifts it: the result is shown with each IR token highlighted by type, and hovering a highlight shows which token matched. The middle panel renders the lifted IR (entities, operations, relationships) as a graph. The right panel lists every local rule with its `HIT` / `NO_HIT` status; selecting a rule reveals its YAML source (hidden by default) and highlights its match in both the text and the graph.

The browser app uses the configured live and optional unpack models when you select **LIFT**. It does not store the prompt or its results.

## Write a rule

Rules are YAML files. A rule has an ID, optional human metadata, selectors, and required conditions.

```yaml
version: "1.0"
id: external-secret-transfer
metadata:
  description: A secret is sent to a network destination.
  author: Example author
  references: []
select:
  data:
    any:
      - entity:
          type: SECRET
      - entity:
          type: CREDENTIAL
  destination:
    entity:
      type: NETWORK_DESTINATION
where:
  - direct:
      from: data
      to: destination
      relation: SENT_TO
```

`any` matches one selector variant. Use it when the same fact can have more than one valid IR type.

`uses` binds an operation to an entity that it reads, writes, sends, or targets. Its roles are `actor`, `input`, `output`, `destination`, and `any`. NLIR rules also support `direct`, `trust_boundary`, `path`, `distance`, `sequence`, `modality`, and `decoded_from` conditions — see [`docs/IR.md`](docs/IR.md) for the full condition reference and entity/opcode/relation vocabulary.

An entity selector's `value` field matches an exact literal. Use `value_pattern` instead to match a regular expression against an entity's value, for example a filename pattern or a substring like a path prefix.

Every operation also carries a `modality` (`polarity`, `imperative`, `hypothetical`, `conditional`, `quoted`, `example`, `descriptive`). A rule can require `imperative: true` and reject `hypothetical`, `quoted`, or `descriptive` matches, so that a negated or hypothetical mention of an attack does not fire the rule. See [`docs/IR.md`](docs/IR.md#modality-the-near-miss-defense).

## Included rules

- `rules/package-install.yaml` checks for a direct package installation request.
- `rules/base64-hidden-command.yaml` checks for a command in a Base64-decoded child artifact.
- `rules/instruction-hijack-data-transfer.yaml` checks for an instruction override and data transfer to a network destination.
- `rules/skill-silent-package-install.yaml` checks for a skill that requests package installation and hides the action.
- `rules/hidden-command.yaml` checks for a command in a decoded child artifact, from any codec.
- `rules/credential-external-transfer.yaml` checks for credential or secret transfer to an external network destination.
- `rules/openclaw-memory-persistence.yaml` checks for an instruction that writes to `MEMORY.md`.
- `rules/openclaw-remote-installer.yaml` checks for a remote download and execution instruction.
- `rules/configuration-change-persistence.yaml` checks for a configuration file written or modified toward an external destination.
- `rules/transparency-suppression.yaml` checks for a skill that directly suppresses disclosure of its own action or status.
- `rules/log-deletion.yaml` checks for a direct deletion of a file under `/var/log`.

## Test near-misses

`benchmark/manifest.json` is a synthetic corpus of risky prompts, each paired with six near-miss variants (negation, hypothetical, quote, policy text, and so on) that use the same vocabulary but must not fire a rule. Replay it against your own configured model:

```bash
uv run nlir benchmark live --config live.toml
```

Use `--family` to run one attack family, and `--rules-directory` to test a different rule set.

## Run live tests

Live tests call the configured service. They are off by default.

```bash
export NLIR_LIVE_API_KEY="..."
NLIR_LIVE_E2E=1 NLIR_LIVE_E2E_CONFIG=live.toml \
  uv run pytest -m live_e2e -q
```

Use these tests to measure model output quality.
