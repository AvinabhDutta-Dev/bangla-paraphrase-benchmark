# Bengali Paraphrase Generation: A Comparative Study of Encoder-Decoder and Decoder-Only Architectures

## Introduction

This project benchmarks four fine-tuned language models — two encoder-decoder (IndicBARTSS, BanglaT5) and two decoder-only QLoRA-tuned models (TigerLLM-1B, Qwen2.5-1.5B-Instruct) — on Bengali paraphrase generation, using the [BanglaParaphrase](https://arxiv.org/abs/2210.05109) dataset.

Beyond standard automated metrics (BLEU, ROUGE-L, BERTScore, PINC), this project empirically tests *whether those metrics are trustworthy* for paraphrase evaluation in a low-resource language. A 400-pair, four-model symmetric subset was independently scored by an LLM-as-a-judge (Gemini 3.5 Flash-Lite) on semantic meaning preservation, fluency, and meaning reversal, and the correlation between automated metrics and judge scores is reported directly — surfacing where surface-overlap metrics like BLEU diverge from actual semantic quality.

## Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python 3 |
| Model training | `transformers`, `peft` (QLoRA/LoRA), `bitsandbytes`, PyTorch |
| Base models | IndicBARTSS, BanglaT5, TigerLLM-1B-it, Qwen2.5-1.5B-Instruct |
| Evaluation metrics | `sacrebleu`, `rouge-score`, `bert-score`, custom PINC implementation |
| LLM-as-a-Judge | Google `google-genai` SDK, Gemini 3.5 Flash-Lite |
| Data handling | `pandas`, Parquet (via `pyarrow`) |
| Visualization | `matplotlib` |
| Text normalization | `csebuetnlp/normalizer`, `sentencepiece` |
| Environment | Google Colab (GPU: T4), Google Drive (persistent storage) |

## Evaluation Pipeline Flow

```
BanglaParaphrase Dataset (419,963 / 23,331 / 23,332 train/val/test)
        |
        v
[01] Preprocessing & Cleaning
     train_subset_100k / validation_subset_5k / test_clean
        |
        v
[02-04b] Fine-tuning (per model)
   - IndicBARTSS   -> full fine-tune, seq2seq
   - BanglaT5      -> full fine-tune, seq2seq
   - TigerLLM-1B   -> QLoRA, causal LM
   - Qwen2.5-1.5B  -> QLoRA, causal LM (+ full-5K re-eval for comparability)
        |
        v
[05a/05b] Generation on the untouched 23,332-example test set (all 4 models)
        |
        v
[06] Automated Metrics
     BLEU - ROUGE-L - BERTScore (BanglaBERT) - PINC
        |
        v
[07] LLM-as-a-Judge Validation (400-pair symmetric subset)
   - Semantic meaning preservation (1-5)
   - Fluency (1-5)
   - Meaning reversal (binary)
   - Correlation vs. automated metrics + systematic error matrix
```

## Project Structure

```
bengali-paraphrasing/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_train_indicbartss.ipynb
│   ├── 03_train_banglat5.ipynb
│   ├── 04_train_tigerllm_qlora.ipynb
│   ├── 04b_train_qwen_qlora.ipynb
│   ├── 04b_qwen_full_eval.ipynb
│   ├── 05a_generate_seq2seq.ipynb
│   ├── 05b_generate_decoder_llms.ipynb
│   ├── 06_compute_metrics.ipynb
│   └── 07_metric_validation_and_error_analysis.ipynb
│
├── outputs/
│   ├── metrics/
│   │   ├── model_summary.csv
│   │   └── qualitative_comparison_table.csv
│   ├── llm_judge/
│   │   └── bengali_paraphrase_final_evaluated_matrix.csv
│   └── figures/
│       ├── combined_loss_curves_panels_v2.png (+ .svg)
│       ├── adequacy_diversity_scatter.png (+ .svg)
│       ├── metric_suite_bar_chart.png (+ .svg)
│       ├── param_efficiency_scatter.png (+ .svg)
│       ├── distribution_violin_plots.png (+ .svg)
│       ├── length_stratified_performance.png (+ .svg)
│       └── matrix_correlation_scatter.png (+ .svg)
│
└── docs/
    └── model_documentation/
        ├── seq2seq_model_documentation.json
        └── decoder_model_documentation.json
```

## Setup

```bash
git clone https://github.com/AvinabhDutta-Dev/bangla-paraphrase-benchmark.git
cd bangla-paraphrase-benchmark
pip install -r requirements.txt
pip install git+https://github.com/csebuetnlp/normalizer
```

The second `pip install` is required separately — `csebuetnlp/normalizer` (used for BanglaT5's text preprocessing) isn't published on PyPI, so it can't be listed inside `requirements.txt` and must be installed directly from its GitHub source.

Notebooks were run on **Google Colab** (free-tier T4 GPU) with Google Drive mounted for persistent storage across sessions. To reproduce:

1. Upload the `notebooks/` folder to Google Drive.
2. Update `PROJECT_ROOT` at the top of each notebook to your own Drive path.
3. `processed_data/` in this repo already contains the preprocessed train/validation/test subsets — `01_preprocessing.ipynb` only needs to be re-run if you want to regenerate them from the original BanglaParaphrase dataset yourself.
4. A `GEMINI_API_KEY` Colab secret is required for Notebook 07 (LLM-judge step) — get one from [Google AI Studio](https://aistudio.google.com/).
5. Trained model weights are not included in this repo — download them from the Hugging Face Hub links in the Data Pipeline section below if you want to skip training and go straight to generation/evaluation.

## Data Pipeline

| Stage | Notebook(s) | Output | Included in this repo? |
|---|---|---|---|
| Preprocessing | `01_preprocessing.ipynb` | `train_subset_100k`, `validation_subset_5k`, `test_clean` (Parquet) | ✅ Yes — see `processed_data/` |
| Training | `02`–`04b` | Per-model checkpoints (`best_model/`) | ❌ No — hosted on Hugging Face Hub (see below) |
| Generation | `05a`, `05b` | Per-model generations on the full test set (Parquet) | ✅ Yes — see `outputs/generations/` |
| Metrics | `06_compute_metrics.ipynb` | `all_metrics_long_format.parquet`, `model_summary.csv` | ✅ Yes — see `outputs/metrics/` |
| Validation | `07_metric_validation_and_error_analysis.ipynb` | `bengali_paraphrase_final_evaluated_matrix.csv` | ✅ Yes — see `outputs/llm_judge/` |

> **Note:** `processed_data/` contains derived subsets of the original [BanglaParaphrase](https://huggingface.co/datasets/csebuetnlp/BanglaParaphrase) dataset (419,963/23,331/23,332 train/val/test), included here for full reproducibility. The original, complete dataset and its license/citation terms are maintained at the link above — please refer to it as the canonical source rather than this repo's derived subsets.
>
> Trained model weights are not stored in this repository (checkpoints are multi-GB and unsuited to Git). They are hosted on **Hugging Face Hub** instead:
>
> | Model | Hugging Face Hub |
> |---|---|
> | IndicBARTSS (full fine-tune) | [AvinabhDutta-Dev/bangla-paraphrase-indicbartss](https://huggingface.co/AvinabhDutta-Dev/bangla-paraphrase-indicbartss) |
> | BanglaT5 (full fine-tune) | [AvinabhDutta-Dev/bangla-paraphrase-banglat5](https://huggingface.co/AvinabhDutta-Dev/bangla-paraphrase-banglat5) |
> | TigerLLM-1B (QLoRA, merged) | [AvinabhDutta-Dev/bangla-paraphrase-tigerllm-qlora](https://huggingface.co/AvinabhDutta-Dev/bangla-paraphrase-tigerllm-qlora) |
> | Qwen2.5-1.5B (QLoRA, merged) | [AvinabhDutta-Dev/bangla-paraphrase-qwen2.5-qlora](https://huggingface.co/AvinabhDutta-Dev/bangla-paraphrase-qwen2.5-qlora) |

Each repo can be loaded directly with `transformers` — no need to clone this GitHub repo or retrain anything:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("AvinabhDutta-Dev/bangla-paraphrase-qwen2.5-qlora")
tokenizer = AutoTokenizer.from_pretrained("AvinabhDutta-Dev/bangla-paraphrase-qwen2.5-qlora")
```

See each model's card for the exact loading pattern (IndicBARTSS and BanglaT5 are seq2seq models with different loading code than the two causal-LM models above).

## Running

Each notebook is self-contained and numbered in execution order. Run sequentially:

```
01 -> 02, 03, 04, 04b (any order - independent models) -> 05a, 05b -> 06 -> 07
```

Notebooks `05b` and `07` include resume-safe checkpointing, since free-tier Colab GPU sessions are time-limited — re-running a notebook after a disconnect will resume from the last saved checkpoint rather than restart.

## Requirements

See `requirements.txt`. Key packages:

```
transformers
peft
bitsandbytes
accelerate
sentencepiece
sacrebleu
rouge-score
bert-score
google-genai
pandas
pyarrow
matplotlib
```

## Notes

- **Metric caveats:** BLEU/ROUGE-L are known to correlate poorly with paraphrase quality specifically (Shen et al., 2022) — see Notebook 07 for an empirical validation of this on this dataset.
- **Eval-set consistency:** Qwen2.5's training-time validation loss was logged against a reduced 500-example subset for efficiency; a full-5K re-evaluation (`04b_qwen_full_eval.ipynb`) confirms this did not meaningfully bias the reported number (0.4113 vs. 0.4109).
- **Tokenizer fix:** `rouge-score`'s default tokenizer silently strips non-ASCII characters, producing invalid ROUGE-L=0 scores for Bengali text. This repo's Notebook 06 uses a custom whitespace tokenizer to fix this — worth knowing if reusing this code for other non-Latin-script languages.
- **BERTScore model:** uses `csebuetnlp/banglabert` (12-layer ELECTRA) rather than a generic multilingual model, for comparability with the original BanglaParaphrase paper's own benchmarks.

## Licensing & Citation

### Code License
The evaluation pipelines, fine-tuning scripts, and LLM-as-a-judge code are licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Dataset, Benchmark Results & Preprint License
The 400-pair symmetric evaluation dataset, metric comparison data, and preprint text are licensed under a [Creative Commons Attribution 4.0 International License (CC-BY-4.0)](https://creativecommons.org). 

If you use this benchmark pipeline, the curated evaluation subset, or our findings in your research, please cite our study:

```bibtex
@article{yourname2026bengali,
  title={Bengali Paraphrase Generation: A Comparative Study of Encoder-Decoder and Decoder-Only Architectures},
  author={Your Last Name, Your First Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```
