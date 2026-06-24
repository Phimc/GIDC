# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Reference implementation of **GIDC** (Ghost Imaging using a Deep neural network Constraint) from Wang et al., *Light Sci Appl* 11, 1 (2022). It reconstructs a super-resolution image from single-pixel ghost-imaging measurements. Academic/non-commercial use only.

## Running

There is no build, test, or lint setup — it is a two-file research script. Run the reconstruction with:

```
python GIDC_main.py
```

This loads `data.mat`, runs the optimization, pops up matplotlib figures every 100 steps, and writes BMPs to `./results/`.

### Environment (pinned, old)
- python 3.6
- **tensorflow 1.9** (TF1.x graph/session API — `tf.placeholder`, `tf.Session`, `tf.variable_scope`; not TF2-compatible)
- numpy 1.18.1, matplotlib 3.1.3, pillow 7.1.2

## Architecture

The key idea: this is **not** a trained network. There is no dataset and no pre-training. Each run optimizes the weights of a randomly-initialized U-Net from scratch against a *single* measurement set, using the known physical forward model as the only supervision. The trained weights are discarded; the reconstructed image is the byproduct.

Forward model: `y = A x`, where `A` = known illumination patterns, `y` = known single-pixel measurements, `x` = unknown object image.

Two files:

- [GIDC_main.py](GIDC_main.py) — driver. Loads data, computes a **DGI** (Differential Ghost Imaging) reconstruction analytically (the loop at lines 59-68), uses that DGI result as the network *input* `inpt`, builds the loss/optimizer, and runs the optimization loop.
- [GIDC_model_Unet.py](GIDC_model_Unet.py) — `inference()` defines a U-Net (encoder conv0–conv5 with stride-2 poolings, decoder conv6–conv10 with transpose-convs + skip-concats) producing `out_x` (sigmoid image), then re-applies the **physical measurement model** in the `measurement` scope: it convolves `out_x` with the patterns `A` to produce `out_y`. So the network outputs both the image estimate and its predicted measurements.

Loss = `mean((y - out_y)^2)` + a tiny total-variation regularizer on `out_x` (`TV_strength`). Only `out_y` vs. measured `y` drives learning — the image `out_x` is constrained only indirectly through the physics and the TV term.

### Data flow
1. `data.mat` provides `patterns` (`64×64×N`) and `measurements` (`N`).
2. `num_patterns = round(64*64*SR)` selects how many patterns/measurements to use (sampling rate `SR`).
3. DGI is computed from those, normalized, and fed as the network input.
4. Inputs (`DGI`, `y_real`, `A_real`) are each z-score normalized before the loop.
5. The U-Net is iterated `Steps` times via Adam; outputs are saved/plotted periodically.

## Key parameters (top of GIDC_main.py)

- `SR` (sampling rate, default 0.1) — fewer patterns = harder reconstruction. Raising it requires `num_patterns <= patterns.shape[-1]` or the script raises "Please set a smaller SR".
- `lr0` (0.05), `Steps` (201), `TV_strength` (1e-9), `img_W`/`img_H` (64, must match the data).

## Gotchas

- Reshapes use Fortran order (`order='F'`) when moving between the network's tensor layout and display/save — preserve this when touching reshape/transpose code or the image will be scrambled/transposed.
- Commented-out lines (e.g. `DGI = np.transpose(DGI)` at line 105, alternate normalizations in the model's `measurement` scope) are documented alternatives the authors note "sometimes give better results" — leave them as hints, don't delete.
- `plt.show()` blocks; the loop only advances after each figure window is closed.
