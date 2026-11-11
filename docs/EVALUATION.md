# Mac verification and evaluation

Run on November 11, 2026 using an Apple Silicon Mac, Python 3.9.6, Ollama 0.40.16, and the local `llama3.2:3b` model. The [raw results](benchmark.json) capture model digest, fixture hashes, per-case predictions, warnings and measured timings.

| Metric | Offline rules | Ollama pipeline |
| --- | ---: | ---: |
| Category accuracy | 22/24 (91.7%) | 22/24 (91.7%) |
| Priority accuracy | 22/24 (91.7%) | 16/24 (66.7%) |
| Category macro-F1 | 0.919 | 0.911 |
| Target policy in top three | 20/24 (83.3%) | 20/24 (83.3%) |
| Target policy cited | 20/24 (83.3%) | 19/24 (79.2%) |
| Median end-to-end latency per case | 0.0004 s | 4.88 s |
| Fallback cases | 0 | 0 |

The evaluator calls the same pipeline as the dashboard. Both engines share retrieval and the Ollama pipeline includes deterministic overrides. The dataset has four synthetic tickets in each category and is stored separately from the 12 inbox samples. Its 24 cases informed refinements to the rules, retrieval, and prompt, so the reported scores describe a development set. The benchmark provides no independent estimate of production accuracy.

**Accuracy** counts exact label matches. **Category macro-F1** averages the F1 of each supported category, giving them equal weight. **Policy hit@3** checks whether a single labeled policy was among the retrieved passages. **Target policy cited** checks if that ID appears in the final list. Neither metric scores the content or entailment of the draft. The offline template lists its retrieved IDs, so its two policy rates coincide. **Latency** measures the local evaluation call from retrieval through output validation, including model loading if it occurs on that request. It excludes UI and database time.

The Ollama pipeline recovered the access category on EVAL-03, where the rules had no strong trigger. It labeled the payroll outage on EVAL-20 as General instead of Software, although the deterministic urgency floor kept it Critical. It predicted Low on all four Medium billing cases. Both engines missed the paraphrased app crash on EVAL-18. The prompt-injection example EVAL-08 retained Billing in both pipelines, while Ollama still missed priority. One crafted example and mocked failure tests provide limited evidence about hostile inputs.

The full-pipeline metrics include any offline fallbacks. The nested `engine_only` report excludes them and discloses the number of completed model cases. That denominator matters when comparing runs with failures. With zero fallbacks, the Ollama pipeline and model-completed metrics use the same rows. The resulting labels still include the rules' overrides.

## Verification

- **Live Mac test.** Analyzed seeded ticket REQ-001 using Ollama in Streamlit. Verified Access, Medium, Identity & Access Team, and policy POL-ACC-01. Edited the reply, approved, and confirmed the exact edit and approval event in SQLite. The browser exposed a draft-submission timing bug, which was fixed by submitting draft and decision in one form and then retested.
- **Automated tests.** Cover the offline pipeline, mocked Ollama response and fallback handling, citation filtering, security overrides, atomic CSV import, duplicate and stale approval handling, and the dashboard review lifecycle with a temporary database.
- **Scope.** The optional live OpenAI path, deployment, real ticket data, reviewer identity, and external actions were outside this run. The evaluator does not score reply quality, measure business outcomes, or certify prompt-injection resistance.

[Local Ollama recommendation](dashboard.jpg) · [Approved edited reply](approved-reply.jpg)
