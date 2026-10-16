# BRISC: measured RIS channels (for experiment e12)

M. Piana, G. A. Alghisi, A. V. Guglielmi, G. Perin, F. Gringoli, S. Tomasin,
"BRISC: A Dataset of Channel Measurements at 5 GHz With a Reflective Intelligent Surface,"
arXiv:2602.21102, 2026. Data: Zenodo, doi:10.5281/zenodo.18714621, **CC BY 4.0**.

* RIS: 16×16 = 256 elements, 1-bit (Open Source RIS board), 5.53 GHz, 242 OFDM subcarriers
* TX at 9 positions, fixed RX (log-periodic and dipole antennas, separate files)
* ~10 000 RIS configurations per position: 16 (2×2 blocks) + 512 (3×3 blocks), then element-wise random

## Download (on the server, ~2.1 GB per position)

```bash
cd data/brisc
bash download_brisc.sh            # log-periodic RX, TX positions 1..9 (~19 GB)
# or fewer: bash download_brisc.sh 1 2 3
```

## Preprocess once (average the frames of each configuration -> cache/*.npz, ~20 MB each)

```bash
cd ../..                         # repository root
python -m mpr.brisc prepare --data data/brisc --antenna log
```

Needs `h5py` and `scipy` (`pip install h5py scipy`). Reads the 2 GB files in chunks,
about 1–2 min per file. After this the .mat files are no longer needed by e12.

## Files used

| file | content |
|---|---|
| `configurations_10000.mat` | 10 000 configurations, hex strings `!0X…` (element 1 = MSB) |
| `antennaLog_pos<k>.mat` | `csi` [242 × frames], `confId` [1 × frames] (1-based) |
| `cache/antennaLog_pos<k>.npz` | written by `prepare`: mean CSI per configuration `H` (10000 × 242), `count` |
| `configurations_bits.npy` | written by `prepare`: configurations as 0/1 array (10000 × 256) |

## Citation

If you use these data, cite the dataset (Zenodo DOI above, CC BY 4.0) and the arXiv paper.
