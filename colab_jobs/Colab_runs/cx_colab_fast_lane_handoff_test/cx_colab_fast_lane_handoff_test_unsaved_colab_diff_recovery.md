# CX Colab Fast-Lane Handoff Test Unsaved Colab Diff Recovery

Source attachment:

```text
C:\Users\jakem\.codex\attachments\1a9ec0ac-0890-4a2c-b119-656c5d312f58\pasted-text.txt
```

Context:

The Colab notebook content failed to save to Drive. The pasted unsaved diff showed the generated capsule content and early execution output from the Colab session.

Recovered facts:

- The unsaved Colab constants cell embedded `execution_scope: full-fidelity simulation`.
- The unsaved Colab constants cell embedded `fidelity_requirement: exact reproduction; scientific source unchanged`.
- The unsaved Colab constants cell embedded `XLA_PYTHON_CLIENT_MEM_FRACTION: 0.90`.
- The runtime setup cell applied these environment overrides before JAX imports:

```json
{
  "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
  "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
}
```

- Google Drive mounted at `/content/drive`.
- The extraction cell reported:

```text
Extracted and verified 9 files in /content/qm_job
```

- The control-panel Markdown table was present with:

```text
T1 -> terminal_T1
T2 -> terminal_T2
T3 -> terminal_T3
J1 -> terminal_J1
```

Resolution:

The local manifest and regenerated notebook have been restored to the intended `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90` memory policy. This is an environment resource policy for Colab A100 usage only. It does not alter scientific source, equations, arguments, or comparison gates.
