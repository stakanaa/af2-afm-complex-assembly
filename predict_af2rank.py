"""MSA-free complex structure prediction with AlphaFold2 and AlphaFold-Multimer.

Each prediction takes a single two-chain template (a docked pose) and no MSA.
The target sequence is read from the template structure itself, and the
template sequence and all side chains other than CB are masked, so the
template contributes backbone and CB geometry only.

The ``af2rank`` class below is taken from the AF2Rank example notebook that
ships with ColabDesign (``af/examples/AF2Rank.ipynb``). Relative to the
notebook:

- the scoring helpers that ranked decoys are gone, and with them the TM-score
  and composite score, because the quantities the paper reports are computed
  from the written structures instead;
- the ``iterations`` loop that fed a prediction back as its own template is
  gone; it did nothing at the default of one iteration;
- ``data_dir`` and ``seed`` are passed through to ColabDesign's own
  parameters, which the notebook does not expose;
- one prediction writes a structure for every recycling iteration, rather than
  one structure at the end.

ColabDesign is installed unmodified, but ``save_recycles.py`` subclasses two
of its methods; see README.md.

Structures are written for recycling iterations 0..3 ("Recycle 0" to
"Recycle 3"), all from one prediction: ColabDesign runs recycling as a
python loop, and ``save_recycles.py`` keeps the output of every pass, which
is how the paper's structures were obtained.

Usage
-----
    python predict_af2rank.py \
        --native    example/1IRA/native.pdb \
        --templates example/1IRA/templates_predicted-monomers \
        --out       output/1IRA_predicted-monomers_afm \
        --model     afm \
        --chain     A,B \
        --data-dir  .

``--data-dir`` is the directory *containing* ``params/``, not ``params/``
itself.
"""

import argparse
import json
import os
import re
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np

from colabdesign import clear_mem
from colabdesign.shared.utils import copy_dict

from save_recycles import mk_af_model

# AF2 is the single-chain network; AFM is the multimer network. The paper uses
# model 1 of each, matching the parameter sets named in the Methods.
MODEL_NAME = {"af2": "model_{n}_ptm", "afm": "model_{n}_multimer_v3"}

# Confidence metrics worth keeping. Both networks report all four for a
# two-chain input, i_ptm included. Everything the paper actually reports
# (inter-chain CB-CB contacts, LRMSD, TM-score, secondary structure, DockQ) is
# computed externally from the written PDB files, not here.
SCORE_KEYS = ("plddt", "ptm", "i_ptm", "pae")


def _atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """Sort "2.pdb" before "10.pdb"."""
    return [_atoi(c) for c in re.split(r"(\d+)", text)]


def evoformer_capture_available():
    """Whether the optional Evoformer-capture patch is applied.

    Without the patch, --save-evoformer raises only after a prediction has
    already run, so this is checked at startup instead. If the sources cannot
    be read, say yes and let the later check decide.
    """
    import inspect

    from colabdesign.af.alphafold.model import modules, modules_multimer

    try:
        return all("output_iteration" in inspect.getsource(m)
                   for m in (modules, modules_multimer))
    except (OSError, TypeError):
        return True


class af2rank:
    """Template-based, MSA-free prediction (Roney & Ovchinnikov, 2022)."""

    def __init__(self, pdb, chain=None, model_name="model_1_ptm",
                 model_names=None, data_dir=".", save_recycles=True):
        self.args = {"pdb": pdb, "chain": chain,
                     "use_multimer": ("multimer" in model_name),
                     "model_name": model_name,
                     "model_names": model_names,
                     "data_dir": data_dir,
                     "save_recycles": save_recycles}
        self.reset()

    def reset(self):
        self.model = mk_af_model(protocol="fixbb",
                                 use_templates=True,
                                 use_multimer=self.args["use_multimer"],
                                 debug=False,
                                 model_names=self.args["model_names"],
                                 data_dir=self.args["data_dir"],
                                 save_recycles=self.args["save_recycles"])

        self.model.prep_inputs(self.args["pdb"], chain=self.args["chain"])
        self.model.set_seq(mode="wildtype")
        self.wt = self.model._wt_aatype

    def set_pdb(self, pdb, chain=None):
        if chain is None:
            chain = self.args["chain"]
        self.model.prep_inputs(pdb, chain=chain)
        self.model.set_seq(mode="wildtype")
        self.wt = self.model._wt_aatype

    def set_seq(self, seq):
        self.model.set_seq(seq=seq)
        self.wt = self.model._params["seq"][0].argmax(-1)

    def predict(self, pdb=None, seq=None, chain=None,
                input_template=True, model_name=None,
                rm_seq=True, rm_sc=True, rm_ic=False,
                recycles=1, output_pdbs=None, extras=None,
                seed=None, save_evoformer=None):
        """Predict from one template; return one score dict per structure.

        ``output_pdbs`` maps a recycling iteration to the file to write it to.
        With ``save_recycles`` on, one prediction covers every iteration up to
        ``recycles``, which is how the paper's structures were produced;
        otherwise only iteration ``recycles`` itself is available.
        """
        if model_name is not None:
            self.args["model_name"] = model_name
            if "multimer" in model_name:
                if not self.args["use_multimer"]:
                    self.args["use_multimer"] = True
                    self.reset()
            else:
                if self.args["use_multimer"]:
                    self.args["use_multimer"] = False
                    self.reset()

        if pdb is not None:
            self.set_pdb(pdb, chain)
        if seq is not None:
            self.set_seq(seq)

        # The target sequence doubles as the template sequence; it is masked
        # out again by rm_template_seq below.
        self.model._inputs["batch"]["aatype"] = self.wt

        # rm_ic=False lets inter-chain template distances through. AlphaFold-
        # Multimer masks them by default because its training templates are
        # retrieved per chain; ColabDesign exposes that mask as this option and
        # leaves it off, so both networks see the same geometry.
        self.model.set_opt(template=dict(rm_ic=rm_ic),
                           num_recycles=recycles)
        self.model._inputs["rm_template"][:] = not input_template
        self.model._inputs["rm_template_sc"][:] = rm_sc
        self.model._inputs["rm_template_seq"][:] = rm_seq

        self.model.predict(models=self.args["model_name"],
                           seed=seed, verbose=False)

        # save_recycles leaves one aux per iteration; without it, only the last
        # iteration exists.
        if self.model.save_recycles:
            auxs = dict(enumerate(self.model.aux["recycles"]))
        else:
            auxs = {recycles: self.model.aux}

        if output_pdbs is None:
            output_pdbs = {}

        scores = []
        for r in (sorted(output_pdbs) if output_pdbs else [recycles]):
            if r not in auxs:
                raise RuntimeError(
                    f"recycling iteration {r} was not kept; this run covers "
                    f"{sorted(auxs)}")
            aux = auxs[r]
            score = {k: float(v) for k, v in copy_dict(aux["log"]).items()
                     if k in SCORE_KEYS}
            if "pae" in score:
                score["pae"] *= 31.0  # stored normalised to [0, 1]
            score["recycles"] = r
            if extras is not None:
                score.update(extras)
            if r in output_pdbs:
                self.model.save_pdb(output_pdbs[r], aux=aux)
            scores.append(score)

        # Per-Evoformer-block pair representations; requires
        # patches/01-evoformer-capture.patch.
        if save_evoformer is not None:
            if "output_iteration" not in self.model.aux:
                raise RuntimeError(
                    "model.aux has no 'output_iteration'. Apply "
                    "patches/01-evoformer-capture.patch to reproduce the "
                    "Evoformer analysis.")
            np.save(save_evoformer, self.model.aux["output_iteration"],
                    allow_pickle=True)

        return scores


def main():
    p = argparse.ArgumentParser(
        description="MSA-free complex prediction from docked templates.")
    p.add_argument("--native", required=True,
                   help="A structure of the complex, parsed once to set the "
                        "model up; each template then replaces it. It need "
                        "not be the experimental structure, but it has to be "
                        "the same complex. With --include-native it is also "
                        "used as a template.")
    p.add_argument("--templates",
                   help="Directory of template PDBs (the docked poses). Only "
                        "files ending in .pdb are read. Optional if "
                        "--include-native is given.")
    p.add_argument("--out", required=True,
                   help="Output directory for predicted structures.")
    p.add_argument("--model", required=True, choices=sorted(MODEL_NAME),
                   help="af2 = model_N_ptm, afm = model_N_multimer_v3.")
    p.add_argument("--chain", default=None,
                   help="Chains to read, comma separated, e.g. A,B. Each one "
                        "is parsed separately, so the comma is required. "
                        "Default: all chains.")
    p.add_argument("--model-num", type=int, default=1, choices=range(1, 6),
                   help="Which of the five parameter sets to use.")
    p.add_argument("--recycles", default="0,1,2,3",
                   help="Recycling iterations to write, comma separated. One "
                        "prediction covers all of them.")
    p.add_argument("--max-poses", type=int, default=100,
                   help="Use only the first this many templates, in ZDOCK rank "
                        "order. The paper uses the 100 top-ranked of 2000 "
                        "poses. Pass 0 for all of them.")
    p.add_argument("--data-dir", default=".",
                   help="Directory holding params/params_<model>.npz.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed, passed to model.predict. Left unset by "
                        "default, as in the runs the paper reports; inference "
                        "here is deterministic, so it changes nothing.")
    p.add_argument("--include-native", action="store_true",
                   help="Also use the structure given as --native as a "
                        "template. With no --templates it is the only one, "
                        "which is how the paper's poly-X runs were done.")
    p.add_argument("--keep-template-seq", action="store_true",
                   help="Do not mask the template sequence (not used in the "
                        "paper).")
    p.add_argument("--keep-template-sidechains", action="store_true",
                   help="Do not mask template side chains beyond CB (not used "
                        "in the paper).")
    p.add_argument("--mask-interchain", action="store_true",
                   help="Mask inter-chain template distances, which restores "
                        "AlphaFold-Multimer's own default. Left off here, as "
                        "in the paper, so that both networks receive the same "
                        "geometry.")
    p.add_argument("--save-evoformer", action="store_true",
                   help="Also save per-Evoformer-block pair representations. "
                        "Requires the optional patch, and a lot of memory.")
    args = p.parse_args()

    if not args.templates and not args.include_native:
        p.error("give --templates, or --include-native to use the structure "
                "given as --native as the only template")

    if args.save_evoformer and not evoformer_capture_available():
        p.error("--save-evoformer needs patches/01-evoformer-capture.patch, "
                "which the installed ColabDesign does not have; see README.md")

    recycles = [int(r) for r in args.recycles.split(",") if r != ""]
    model_name = MODEL_NAME[args.model].format(n=args.model_num)
    os.makedirs(args.out, exist_ok=True)

    settings = dict(rm_seq=not args.keep_template_seq,
                    rm_sc=not args.keep_template_sidechains,
                    rm_ic=args.mask_interchain,
                    seed=args.seed)

    clear_mem()
    af = af2rank(args.native, chain=args.chain, model_name=model_name,
                 data_dir=args.data_dir)

    # Templates are taken in ZDOCK rank order, which the file names encode.
    poses = []
    if args.templates:
        poses = sorted((name for name in os.listdir(args.templates)
                        if name.endswith(".pdb")), key=natural_keys)
        if args.max_poses:
            poses = poses[:args.max_poses]
    templates = [("native.pdb", args.native)] if args.include_native else []
    templates += [(name, os.path.join(args.templates, name)) for name in poses]
    print(f"{len(templates)} template(s), recycles {recycles}, {model_name}")

    scores = []
    for name, path in templates:
        stem = name[:-4] if name.endswith(".pdb") else name
        out_pdbs = {r: os.path.join(args.out, f"{stem}_r{r}.pdb")
                    for r in recycles}

        # One prediction keeps every iteration up to the last one asked for,
        # which is how the paper's structures were obtained. The Evoformer
        # representations are those of that run's last pass.
        num_recycles = max(recycles)
        evof = (os.path.join(args.out, f"{stem}_rep_r{num_recycles}.npy")
                if args.save_evoformer else None)
        got = af.predict(pdb=path, recycles=num_recycles, output_pdbs=out_pdbs,
                         extras={"id": name}, save_evoformer=evof, **settings)
        scores.extend(got)
        for score in got:
            print(f"{stem} r{score['recycles']} " + " ".join(
                f"{k} {score[k]:.4f}" for k in SCORE_KEYS if k in score),
                flush=True)

    with open(os.path.join(args.out, f"scores_{model_name}.json"), "w") as f:
        json.dump(scores, f, indent=2)


if __name__ == "__main__":
    main()
