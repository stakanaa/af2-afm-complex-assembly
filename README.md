# MSA-free complex prediction with AlphaFold2 and AlphaFold-Multimer

Code accompanying:

> Seiya Tanaka and George Chikenji,
> *Revealing the contrasting use of structural information in protein complex
> assembly by AlphaFold2 and AlphaFold-Multimer*,
> Biophysics and Physicobiology (submitted).

## Provenance

The `af2rank` class in `predict_af2rank.py` is taken from the AF2Rank example
notebook distributed with [ColabDesign](https://github.com/sokrypton/ColabDesign)
(`af/examples/AF2Rank.ipynb`), which implements the protocol of
[Roney & Ovchinnikov (2022)](https://doi.org/10.1103/PhysRevLett.129.238101),
and modified here; the head of the file lists how.

ColabDesign is installed unmodified, but it does not behave as it ships:
`save_recycles.py` subclasses `mk_af_model` and overrides two of its methods so
that the structure of every recycling iteration is kept, which is how the
paper's structures were produced. The file says which lines it reproduces from
v1.1.3 and which it adds; nothing in the computation changes.

Licensed under Apache-2.0; `NOTICE` says which parts come from elsewhere and
under what terms. The AlphaFold2 and AlphaFold-Multimer parameters are not
redistributed here; see Setup.

## Setup

```bash
conda env create -f environment.yml
conda activate af2rank
pip install git+https://github.com/sokrypton/ColabDesign.git@v1.1.3
```

The environment is named `af2rank`; pass `-n <name>` or `-p <path>` to
`conda env create` if that name is already taken.

The pinned `jaxlib` is the CUDA 12 / cuDNN 8.9 build, which is not on PyPI, so
`environment.yml` adds Google's wheel index with `--find-links`. That build
links against a system CUDA installation, so CUDA 12 and cuDNN 8.9 have to be
present; it was run here against CUDA 12.2 and cuDNN 8.9.7. Check that the GPU
is visible before going further:

```bash
python -c "import jax; print(jax.devices())"
```

It should print a GPU, as in `[gpu(id=0)]`; a CPU-only list means the CUDA
build is not in use.

Then fetch the network parameters (5.3 GB):

```bash
mkdir -p params
curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar | tar x -C params
```

## Quick start

`example/` holds two targets: 1IRA, with the 100 top-ranked ZDOCK poses used as
templates, and 1R8S, with the single pose the Evoformer analysis uses (see
`example/README.md`).

Start with one template and no recycling, which is a single forward pass:

```bash
python predict_af2rank.py \
    --native    example/1IRA/native.pdb \
    --templates example/1IRA/templates_predicted-monomers \
    --out       output/smoke \
    --model     afm \
    --chain     A,B \
    --data-dir  . \
    --max-poses 1 \
    --recycles  0
```

That writes the predicted complex, `output/smoke/complex.1_r0.pdb`, and
`output/smoke/scores_model_1_multimer_v3.json`, which holds one entry per
structure with its pLDDT, pTM, ipTM and PAE. Those two files appearing,
together with a GPU in the device list above, is what tells you the setup is
complete.

Dropping the last two flags runs all 100 templates at every recycling
iteration.

```bash
python predict_af2rank.py \
    --native    example/1IRA/native.pdb \
    --templates example/1IRA/templates_predicted-monomers \
    --out       output/1IRA_predicted-monomers_afm \
    --model     afm \
    --chain     A,B \
    --data-dir  .
```

That writes `<pose>_r0.pdb` … `<pose>_r3.pdb` for each template.

## Options

`predict_af2rank.py --help` lists them all; these are the ones that matter.

| Option | What it does |
| --- | --- |
| `--native` | A structure of the complex, parsed once to set the model up: its chain IDs and their lengths. It need not be the experimental structure. |
| `--out` | Where to write the structures and the scores. |
| `--templates` | Directory of template PDBs. They are used in file-name order, which for `complex.N.pdb` is ZDOCK rank order. |
| `--max-poses` | Stop after the first this many templates, in that order. Default 100, `0` for all of them. |
| `--recycles` | Which recycling iterations to write. Default `0,1,2,3`. |
| `--model` | `afm` for AlphaFold-Multimer (`model_1_multimer_v3`), `af2` for AlphaFold2 (`model_1_ptm`). |
| `--chain` | Which chains of each structure to read, comma separated. |
| `--data-dir` | The directory that contains `params/`. |
| `--include-native` | Also predict from `--native` as a template. |
| `--mask-interchain` | Masks inter-chain template distances, restoring AlphaFold-Multimer's own default. Left off unless given, as in the paper, so that both networks see the same geometry. |
| `--save-evoformer` | Also save the Evoformer representations. Needs the patch below. |

The target sequence is parsed from the structure file given as `--templates`
and passed to the network as the query sequence, separately from the template
features. There is no separate sequence input, so running a different target
sequence means supplying a template whose residues are that sequence: the
paper's poly-X runs used templates whose side chains had been rebuilt onto a
poly-X sequence with SCWRL4, leaving the backbone unchanged.

## How Recycle 0–3 are obtained

The structures after the initial pass and after each of three recycling
iterations all come from one prediction. Recycling is a plain python loop in
ColabDesign v1.1.3: with the default `recycle_mode` of "last" the model is
called once per iteration and only the last output is kept, so every iteration
is already computed and `save_recycles.py` keeps what would otherwise be
dropped. One prediction with `num_recycles` set to 3 writes all four structures
in four forward passes.

## Optional patch: Evoformer representations

`patches/01-evoformer-capture.patch` applies to ColabDesign v1.1.3 and makes
the representations after each of the 48 Evoformer blocks available to
`--save-evoformer`:

```bash
patch -p1 -d "$(python -c 'import colabdesign,os;print(os.path.dirname(os.path.dirname(colabdesign.__file__)))')" \
      < patches/01-evoformer-capture.patch
```

Then predict with `--save-evoformer`:

```bash
python predict_af2rank.py \
    --native    example/1R8S/native.pdb \
    --templates example/1R8S/templates_predicted-monomers \
    --out       output/1R8S_evoformer \
    --model     afm \
    --chain     A,B \
    --data-dir  . \
    --recycles  0 \
    --save-evoformer
```

`--save-evoformer` writes one `.npy` beside the structure, for the run's last
iteration, holding both representations as a dict:

```python
rep = np.load("complex.100_rep_r0.npy", allow_pickle=True).item()
rep["pair"]   # (48, L, L, 128)
rep["msa"]    # (48, L, 256)
```

While the patch is applied, every prediction keeps all 48 blocks, whether or
not `--save-evoformer` is given, which is memory-hungry. Undo it with the same
command and `-R`:

```bash
patch -R -p1 -d "$(python -c 'import colabdesign,os;print(os.path.dirname(os.path.dirname(colabdesign.__file__)))')" \
      < patches/01-evoformer-capture.patch
```

## External tools

This code needs ColabDesign v1.1.3 and the AlphaFold parameters of 2022-12-06:
`model_1_ptm` for [AlphaFold2](https://doi.org/10.1038/s41586-021-03819-2), and
`model_1_multimer_v3` for
[AlphaFold-Multimer](https://doi.org/10.1101/2021.10.04.463034). The templates
were built with LocalColabFold, ZDOCK 3.0.2 and SCWRL4 (see
`example/README.md`). The targets are the heterodimers of the
[Protein–Protein Docking Benchmark 5.5](https://zlab.umassmed.edu/benchmark/).
