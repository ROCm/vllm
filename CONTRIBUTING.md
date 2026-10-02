# Contributing to ROCm/vllm

This repository is AMD's ROCm fork of [vLLM](https://github.com/vllm-project/vllm).
Most development happens upstream. Changes that are not specific to ROCm should go
to upstream vLLM; see the
[vLLM contributing guide](https://docs.vllm.ai/en/latest/contributing).

______________________________________________________________________

> **Security vulnerabilities**: do not open a public GitHub Issue. See [SECURITY.md](SECURITY.md) for the private reporting process.

______________________________________________________________________

## Developer policies

These policies apply to all forms of activity and engagement in this project.

> [!IMPORTANT]
> AMD employees must also follow the ROCm open source software
> contributing policies at http://u.amd.com/rocm-oss-policies.

### Governance

This project is covered by the
[ROCm Project Governance](https://github.com/ROCm/ROCm/blob/develop/GOVERNANCE.md),
which also defines the code of conduct.

### Licensing

Code contributions to this project are covered under the terms of the
[LICENSE](LICENSE) file (Apache 2.0).

______________________________________________________________________

## Development workflows

### Issue tracking

Before filing a new issue, search through
[existing issues](https://github.com/ROCm/vllm/issues) and
[upstream vLLM issues](https://github.com/vllm-project/vllm/issues) to avoid duplicates.

Provide as much information as possible: command output, GPU model, ROCm version,
vLLM version or commit, and OS version.

### Code style

Follow the upstream vLLM code style and run its `pre-commit` hooks before submitting.

### Pull Requests

All contributions should be submitted through a Pull Request (PR) against the
appropriate branch.

Before submitting a PR:

- Run all applicable tests and validate the results.
- Update documentation as needed.
- Ensure all required GitHub Actions checks pass.

PRs require:

- Approval from the appropriate CODEOWNERS.
- Successful completion of required status checks.
- Compliance with repository security and quality requirements.

### Security Requirements

Contributors must not:

- Commit secrets, tokens, passwords, or credentials.
- Introduce vulnerable dependencies without justification.
- Bypass security controls or required security reviews.

All contributions may be subject to automated security scanning.
