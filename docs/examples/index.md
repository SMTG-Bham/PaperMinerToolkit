# Example notebooks

The notebooks are runnable templates and are rendered without execution during documentation builds. They contain no saved outputs or credentials. Copy one, set paths and model identifiers for your environment, and run only the stages you need.

## Extraction providers

- {doc}`openai_gpt_workflow` configures hosted OpenAI text and vision profiles and runs the complete corpus-to-CSV workflow.
- {doc}`anthropic_claude_workflow` does the same with Anthropic Messages models.
- {doc}`qwen_vllm_workflow` starts an OpenAI-compatible local vLLM server before extraction.

## LDA analysis

- {doc}`lda_model_selection` prepares abstracts, compares topic counts and seeds, trains the selected model, and assigns manual names.
- {doc}`lda_trends` predicts with the fixed model and produces annual, block, and rolling trend plots.
- {doc}`lda_filtering` stores model scores and combines topic and regex filters.

Shell cells intentionally use the installed `pm_*` commands: this makes notebook steps directly transferable to terminals and scheduler scripts.

```{toctree}
:maxdepth: 1

openai_gpt_workflow
anthropic_claude_workflow
qwen_vllm_workflow
lda_model_selection
lda_trends
lda_filtering
```
