# Security Policy

## Supported versions

Security fixes land on the latest released version. Please upgrade to a supported
release before reporting.

| Version | Supported |
| ------- | --------- |
| 1.0.x   | yes       |
| < 1.0   | no        |

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue. Use GitHub's
[private vulnerability reporting](https://github.com/devin-lai/onnx2coreml/security/advisories/new),
or email **markauto75@gmail.com**.

Include a description of the issue, affected versions, and a minimal reproducer if you
have one. You can expect an initial response within a few days.

## Scope and threat model

onnx2coreml loads caller-supplied ONNX models and converts them to Core ML. Treat ONNX
files as untrusted input: a malformed or hostile model could attempt path traversal via
external-data references or trigger excessive resource use. The project pins
`onnx>=1.16`, which carries the external-data path-traversal hardening, and validates
models at load time. Even so, only convert models you obtained from a source you trust,
and run conversion of untrusted models in a sandbox.
