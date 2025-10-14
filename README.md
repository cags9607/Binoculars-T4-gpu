# <img src="./assets/bino-logo.svg" width=40 style="padding-top: 0px"/>  Binoculars: Zero-Shot Detection of LLM-Generated Text [[paper]](https://arxiv.org/abs/2401.12070)[[demo]](https://huggingface.co/spaces/tomg-group-umd/Binoculars)

<p align="center">
  <img src="assets/binoculars.jpg" width="300" height="300" alt="ool Binoculars with Falcon on Top">
</p>

Binoculars is a state-of-the-art method for detecting AI-generated text. Binoculars is a
zero-shot and domain-agnostic (requires no training data) method. It is based on a simple idea: most
decoder-only, causal language models have a huge overlap in pretraining datasets, for e.g. Common Crawl, Pile, etc.
More details about the method and results can be found in the paper **Spotting LLMs with Binoculars: Zero-Shot
Detection of Machine-Generated Text**.

## Additions

In order to fit the models on lower-end GPUs like T4, this repository incorporates the following:

- 4-bit loading (NF4) for both models (observer and performer). This significantly reduces the VRAM usage, allowing both models to fit on the GPU(s) without encountering out-of-memory problems. By doing so, we also leave enough memory to make batch inference more optimal.
- Keep models in memory, but run them sequentially to maintain peak memory usage low.
- Device mapping to move performer logits to CPU, freeing VRAM.
- CPU offload by moving logits to the CPU and emptying the cache. 
- Make use of unshifted logits.
- Shorter max tokens.
- Shared tokenizer between performer and observer.
- Stick to only inferences. No gradients.

## Getting Started

### Installation

To run the implementation of Binoculars, you can clone this repository and install the package using pip. This code was developed and tested with Python 3.9. To install the package, run the
following commands:

```bash
$ git clone https://github.com/deepsee-code/ai-gen-text-classifier.git
$ cd ai-gen-text-classifier
$ pip install -e .
```



### Usage

The model outputs a Binocular score that can be used as a ranking score for human likeliness (the higher the score, the more likely the text is human-generated). The default comes with two thresholds to flag a text as ai-generated if the score is below that threshold: one that maximizes `accuracy` and one that minimizez the false positive rate on a validation set. Note: these thresholds are only valid if the backbone models are HF "tiiuae/falcon-7b" for the observer, and "tiiuae/falcon-7b-instruct" for the performer. 

```
ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
FPR_THRESHOLD = 0.8536432310785527  # optimized for low-fpr (chosen at 0.01%)
```

`FPR_THRESHOLD` achieves `1.02%` false positive rate on a dataset of texts (as in the content_metrics superset table) crawled from high-reputation media outlets (reuters.com, bbc.com, etc.). If we focus our attention only on article-like texts, the false postive rate becomes `0.15%`. For `ACCURACY_THRESHOLD`, the false positive rates are `4.76%` and `4.3%` for all texts and articles, respectively.


To detect AI-generated text, please use the following code snippet

```python
from experimental.falcon_t4_binoculars import FalconT4Binoculars
bino = FalconT4Binoculars(mode = "low-fpr", max_token_observed = 384)

# or if you are on A100 GPU:
# from experimental.falcon_a100_binoculars import FalconA100Binoculars
# bino = FalconA100Binoculars(mode = "low-fpr", max_token_observed = 384)

# ChatGPT (GPT-4) output when prompted with “Can you write a few sentences about a capybara that is an astrophysicist?"

sample_string = '''Dr. Capy Cosmos, a capybara unlike any other, astounded the scientific community with his 
groundbreaking research in astrophysics. With his keen sense of observation and unparalleled ability to interpret 
cosmic data, he uncovered new insights into the mysteries of black holes and the origins of the universe. As he 
peered through telescopes with his large, round eyes, fellow researchers often remarked that it seemed as if the 
stars themselves whispered their secrets directly to him. Dr. Cosmos not only became a beacon of inspiration to 
aspiring scientists but also proved that intellect and innovation can be found in the most unexpected of creatures.'''

print(bino.compute_score(sample_string))  # 0.7566666603088379
print(bino.predict(sample_string))  # 'Most likely AI-Generated'
```

In the above code, user can also pass a `list` of `str` to `compute_score` and `predict` methods to get results for
the entire batch of samples.

#### Batch prediction

First, install dependencies.

```
# Python 3.10+ recommended
pip install -r requirements.txt
```

For usage:

```
python inference.py demo.csv \
  --text-col text \            # name of the column with text (defaults to first column)
  --batch-size 8 \               # adjust for VRAM/throughput
  --mode low-fpr \                # or: accuracy
  --max-len 384 \                 # token limit observed per text
  --output-csv text_predictions.csv
```

What the flags do

- --mode: low-fpr → conservative (fewer false positives); accuracy → overall accuracy-optimized

- --max-len: truncate to this many tokens before scoring.

- --batch-size: higher is faster but uses more VRAM.

### Library

You can also install the package.

```
pip install "git+https://github.com/deepsee-code/ai-gen-text-classifier@main"
```

Then, you can import and make predictions using:

```python
from ai_gen_text import detect_batch

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "¡Hola mundo! Aquí el Dr. Capybara dándole caña al estudio de la astrofísica.",
    '''Dr. Capy Cosmos, a capybara unlike any other, astounded the scientific community with his
groundbreaking research in astrophysics. With his keen sense of observation and unparalleled ability to interpret
cosmic data, he uncovered new insights into the mysteries of black holes and the origins of the universe. As he
peered through telescopes with his large, round eyes, fellow researchers often remarked that it seemed as if the
stars themselves whispered their secrets directly to him. Dr. Cosmos not only became a beacon of inspiration to
aspiring scientists but also proved that intellect and innovation can be found in the most unexpected of creatures.''',
    ""]

res = detect_batch(
    texts,
    device = "t4",
    mode = "accuracy",
    max_len = 384,
    batch_size = 8,  
    progress = True,
    return_text = True
)

print(res)

# [{'text': 'The quick...', 'score': 0.73, 'label': 1, 'mode': 'low-fpr', 'threshold': 0.8536}, ...]
```


### Demo

There is a demo available to predict AI-generated text interactively with a simple UI
using [gradio](https://github.com/gradio-app/gradio). You can run the demo using the following command:

```bash
$ python app.py
```

## Limitations

All AI-generated text detectors aim for accuracy, but none are perfect and can have multiple failure modes (e.g.,
Binoculars is more proficient in detecting English language text compared to other languages).

## Citations

```bibtex
@misc{hans2024spotting,
      title={Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text}, 
      author={Abhimanyu Hans and Avi Schwarzschild and Valeriia Cherepanova and Hamid Kazemi and Aniruddha Saha and Micah Goldblum and Jonas Geiping and Tom Goldstein},
      year={2024},
      eprint={2401.12070},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


