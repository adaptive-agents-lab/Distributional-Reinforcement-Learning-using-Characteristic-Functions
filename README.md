# $\varphi$TD: Distributional Reinforcement Learning using Characteristic Functions

<h5 align="center">
Thomas Mousseau &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Tyler Kastner &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Davide Baldelli<br><br>
Amir-massoud Farahmand
</h5>

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/adaptive-agents-lab/Distributional-Reinforcement-Learning-using-Characteristic-Functions.git
cd Distributional-Reinforcement-Learning-using-Characteristic-Functions
uv sync --locked
```

Python 3.11. Linux uses JAX with CUDA 12

## Paper figures

Generate both figures from the included experiment curves, without an account or network access:

```bash
uv run python plot.py
```

Outputs: `figures/minatar.pdf` and `figures/craftax.pdf`.
The files in `data/` contain the seed curves and their run provenance.

## Training

Train all methods and seeds for each experiment, then plot your runs:

```bash
uv run python train.py minatar
uv run python train.py craftax
uv run python plot.py --data-dir results
```

The settings in `configs/` match the recorded paper runs. Full experiments require substantial GPU time. Add `--smoke` for a short execution check; smoke outputs stay separate from full results.

## Contextual bandit

Train the distribution models and generate the analysis figures:

```bash
uv run python bandit.py
```

Outputs: `figures/bandit.pdf` and `figures/parseval.pdf`. Settings: `configs/bandit.yaml`. Add `--smoke` for a short check.

## Citation

```bibtex
@inproceedings{mousseau2026phitd,
  title = {{$\varphi$TD}: Distributional Reinforcement Learning using Characteristic Functions},
  author = {Mousseau, Thomas and Kastner, Tyler and Baldelli, Davide and Farahmand, Amir-massoud},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026}
}
```
