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

### Environment
- python 3.6+
- **pytorch 1.7+** (uses `nn.Module`, autograd, `torch.optim.Adam`; runs on CPU or CUDA — the script auto-selects via `torch.cuda.is_available()`)
- numpy, scipy, matplotlib, pillow

## Architecture

The key idea: this is **not** a trained network. There is no dataset and no pre-training. Each run optimizes the weights of a randomly-initialized U-Net from scratch against a *single* measurement set, using the known physical forward model as the only supervision. The trained weights are discarded; the reconstructed image is the byproduct.

Forward model: `y = A x`, where `A` = known illumination patterns, `y` = known single-pixel measurements, `x` = unknown object image.

Two files:

- [GIDC_main.py](GIDC_main.py) — driver. Loads data, computes a **DGI** (Differential Ghost Imaging) reconstruction analytically (the loop at lines ~53-68), uses that DGI result as the network *input* `inpt`, builds the loss/optimizer, and runs the optimization loop.
- [GIDC_model_Unet.py](GIDC_model_Unet.py) — `GIDCUNet` (a `torch.nn.Module`). The U-Net (encoder `conv0`–`conv5` with stride-2 down-sampling, decoder `up6`–`conv10` with transpose-convs + skip-concats) produces `out_x` (sigmoid image). Its `forward` then re-applies the **physical measurement model**: it convolves `out_x` with the patterns (`F.conv2d`, full-image cross-correlation) to produce `out_y`. So the network outputs both the image estimate and its predicted measurements.

Loss = `mse(y, out_y)` + a tiny total-variation regularizer on `out_x` (`TV_strength`). Only `out_y` vs. measured `y` drives learning — the image `out_x` is constrained only indirectly through the physics and the TV term.

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

- Display/save orientation: the network works in NCHW and the image is transposed (`.T`) before plotting/saving to reproduce the original TF code's Fortran-order reshape (for a square image, a C-order flatten reshaped Fortran-order equals a transpose). Keep `out_x` and `patterns` in the same `[W, H]` spatial layout or the measurement correlation breaks.
- Commented-out lines (e.g. `DGI = DGI.T`, the alternate `# DGI[DGI<0] = 0`) are documented alternatives the authors note "sometimes give better results" — leave them as hints, don't delete.
- `plt.show()` blocks; the loop only advances after each figure window is closed.
- BatchNorm stays in train mode for the whole run (`model.train()`), matching the original which always fed `isTrain=True` — do not switch to `eval()` for the periodic readouts.
