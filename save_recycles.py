"""ColabDesign's prediction, with the structure of every recycle kept.

The paper reports the predicted structure after the initial pass and after each
of three recycling iterations (Recycle 0 to Recycle 3). Those structures came
from a single prediction in which the output of every iteration was retained,
and this file restores that behaviour.

Recycling is a plain python loop in ColabDesign v1.1.3: with the default
``recycle_mode`` of "last", ``_af_design._recycle`` calls ``_single`` once per
iteration and keeps only the aux of the last one. Every iteration is therefore
already computed, and nothing here changes the computation -- it only keeps
what would otherwise be dropped. One prediction with ``num_recycles`` set to 3
yields all four structures in four forward passes.

ColabDesign itself is installed unmodified; two of its methods are subclassed:

- ``_single`` collects the aux of each pass (four added lines);
- ``run`` groups those passes by iteration and combines each group the way
  ``_af_design.run`` combines its own, so that every iteration's aux can be
  given to ``save_pdb`` and read for scores.

``_combine`` reproduces about a dozen lines of ``_af_design.run`` from v1.1.3,
marked below. It leaves out the sequence-recovery term of that log, which only
the design protocols use. If a later ColabDesign changes how recycling is
structured, the checks in ``run`` raise rather than write wrong structures.
"""

import jax
import numpy as np

from colabdesign.af.model import mk_af_model as _upstream_mk_af_model
from colabdesign.shared.utils import to_float

# The recycle modes that loop in python. "backprop" and "add_prev" compile
# recycling into the model, leaving no per-pass aux to collect.
LOOPED_RECYCLE_MODES = ("average", "first", "last", "sample")


class mk_af_model(_upstream_mk_af_model):
    """``mk_af_model`` that can keep the aux of every recycling iteration.

    With ``save_recycles=False`` (the default) it behaves exactly as upstream.
    With it on, ``run`` leaves the per-iteration aux in ``aux["recycles"]``,
    indexed by iteration, each one shaped like ``aux`` itself.
    """

    def __init__(self, *args, **kwargs):
        # Taken out before super(), which passes **kwargs on to set_args.
        self.save_recycles = kwargs.pop("save_recycles", False)
        self._recycle_aux = []
        super().__init__(*args, **kwargs)

    def _single(self, model_params, backprop=True):
        aux = super()._single(model_params, backprop)
        if self.save_recycles:
            self._recycle_aux.append(aux)
        return aux

    def run(self, *args, **kwargs):
        if not self.save_recycles:
            return super().run(*args, **kwargs)

        mode = self._args["recycle_mode"]
        if mode not in LOOPED_RECYCLE_MODES:
            raise RuntimeError(
                f"save_recycles needs one of the recycle modes that loop in "
                f"python {LOOPED_RECYCLE_MODES}, not {mode!r}, which compiles "
                f"recycling into the model and leaves no per-pass output.")

        self._recycle_aux = []
        out = super().run(*args, **kwargs)
        passes, self._recycle_aux = self._recycle_aux, []

        # run() records in the log which models it used.
        model_nums = self.aux["log"]["models"]
        cycles = int(self.aux["num_recycles"]) + 1
        if len(passes) != len(model_nums) * cycles:
            raise RuntimeError(
                f"expected {len(model_nums)} model(s) x {cycles} pass(es) = "
                f"{len(model_nums) * cycles} passes, collected {len(passes)}. "
                f"The recycling loop of this ColabDesign is not the one "
                f"save_recycles.py was written against (v1.1.3).")

        # The passes arrive grouped by model, in iteration order within each
        # model; regroup them by iteration.
        by_model = [passes[i * cycles:(i + 1) * cycles]
                    for i in range(len(model_nums))]
        self.aux["recycles"] = [
            self._combine([m[r] for m in by_model], model_nums, r)
            for r in range(cycles)]
        return out

    def _combine(self, auxs, model_nums, num_recycles):
        """Combine one iteration's aux across models, as ``run`` does its own."""
        auxs = jax.tree_map(lambda *x: np.stack(x), *auxs)

        def avg_or_first(x):                                    # from run()
            if np.issubdtype(x.dtype, np.integer):
                return x[0]
            else:
                return x.mean(0)

        aux = jax.tree_map(avg_or_first, auxs)                  # from run()
        aux["atom_positions"] = auxs["atom_positions"][0]       # from run()
        aux["all"] = auxs                                       # from run()
        aux["num_recycles"] = num_recycles

        aux["log"] = {**aux["losses"]}                          # from run()
        aux["log"]["plddt"] = 1 - aux["log"]["plddt"]           # from run()
        for k in ["loss", "i_ptm", "ptm"]:                      # from run()
            if k in aux:
                aux["log"][k] = aux[k]
        for k in ["hard", "soft", "temp"]:                      # from run()
            aux["log"][k] = self.opt[k]
        aux["log"] = to_float(aux["log"])                       # from run()
        aux["log"].update({"recycles": num_recycles,            # from run()
                           "models": model_nums})
        return aux
