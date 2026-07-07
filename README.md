# Classic HDF5 Robust Cross-Modal Hashing

This package is an isolated replacement experiment route. It does not import
or modify the existing CLIP/token pipeline.

## 1. Server environment

```bash
cd /path/to/ARDB
python3 -m venv .venv
source .venv/bin/activate
pip install -r classic_hashing/requirements.txt
```

## 2. Convert MIRFlickr MAT files

```bash
python3 -m classic_hashing.data.prepare_mirflickr \
  --image-mat /home/jiyonghui/wangyingchao/classic_hashing/data/mirflickr/mirflickr25k-iall-vgg.mat \
  --text-mat /home/jiyonghui/wangyingchao/classic_hashing/data/mirflickr/mirflickr25k-yall.mat \
  --label-mat /home/jiyonghui/wangyingchao/classic_hashing/data/mirflickr/mirflickr25k-lall.mat \
  --output classic_hashing/artifacts/mirflickr/mirflickr.h5 \
  --split-output classic_hashing/artifacts/mirflickr/split_seed42.npz
```

The split is fixed to the MGSH protocol:

- Query: rows `0:1900`
- Train: rows `1900:11400`
- Database: rows `1900:end`

## 3. Generate label noise

Run each required rate with seed 42:

```bash
python3 -m classic_hashing.data.generate_noise \
  --data-h5 classic_hashing/artifacts/mirflickr/mirflickr.h5 \
  --noise-rate 0.2 --seed 42 \
  --output classic_hashing/artifacts/mirflickr/labels_noise_0.2_seed42.h5

python3 -m classic_hashing.data.generate_noise \
  --data-h5 classic_hashing/artifacts/mirflickr/mirflickr.h5 \
  --noise-rate 0.5 --seed 42 \
  --output classic_hashing/artifacts/mirflickr/labels_noise_0.5_seed42.h5

python3 -m classic_hashing.data.generate_noise \
  --data-h5 classic_hashing/artifacts/mirflickr/mirflickr.h5 \
  --noise-rate 0.8 --seed 42 \
  --output classic_hashing/artifacts/mirflickr/labels_noise_0.8_seed42.h5
```

## 4. Configure and train

Update `dataset.h5_path`, `dataset.noise_path`, `dataset.noise_rate`, and
`model.hash_bits` in the selected YAML file, then run:

```bash
python3 -m classic_hashing.train \
  --config classic_hashing/configs/full.yaml
```

Available independently runnable presets:

- `baseline.yaml`
- `center.yaml`
- `small_loss.yaml`
- `ema.yaml`
- `small_loss_ema.yaml`
- `full.yaml`

Use noise rates `0.2`, `0.5`, `0.8` and hash lengths
`16`, `32`, `64`, `128`. Keep seed 42 and all other MGSH protocol settings
unchanged.

## 5. Optional PR curves

Set `evaluation.plot_pr_curve: true` to generate PR files once after the best
checkpoint is restored. The default is `false`.

To generate them later from a checkpoint:

```bash
python3 -m classic_hashing.evaluation.plot_pr_curve \
  --config classic_hashing/configs/full.yaml \
  --checkpoint classic_hashing/outputs/mirflickr_full/best.pth
```

Outputs:

- `pr_i2t.png`
- `pr_t2i.png`
- `pr_average.png`
- `pr_data.npz`

Every training run writes concise epoch losses, selection diagnostics (only
when Small-loss is enabled), MAP results, and the best result to
`<output_dir>/<experiment>/train.log`.

## 6. Verification

```bash
python3 -m pytest classic_hashing/tests -q
python3 -m compileall -q classic_hashing
```
