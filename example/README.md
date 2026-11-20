# Example input

Two benchmark targets.

```
example/
  1IRA/
    native.pdb                       the experimental complex (PDB 1IRA),
                                     456 residues, chain A 311 and chain B 145
    templates_predicted-monomers/    the 100 top-ranked ZDOCK poses
  1R8S/
    native.pdb                       the experimental complex (PDB 1R8S),
                                     347 residues, chain A 160 and chain B 187
    templates_predicted-monomers/    complex.100.pdb, the pose the Evoformer
                                     analysis uses
```

A directory is named for one target, and the templates for how they were built:
`templates_predicted-monomers` holds poses obtained by docking monomers taken
from an AlphaFold-Multimer prediction of that target. The full dataset uses the
same names with the nesting the other way round, since it holds all 137
targets: `templates_predicted-monomers/1IRA/` there is
`example/1IRA/templates_predicted-monomers/` here.

For 1IRA the templates are the top 100 of the 2000 poses ZDOCK produces, in
rank order (`complex.1.pdb` ... `complex.100.pdb`). For 1R8S only
`complex.100.pdb` is here, the pose the Evoformer analysis was run on; 1IRA is
large enough that keeping all 48 Evoformer blocks for it exhausts the memory of
the card the paper used.

Running all 100 templates takes a while, so raise `--max-poses` gradually.

## Checking the setup

The smoke test in the README writes one structure and one entry of scores,
which for `complex.1` of 1IRA at Recycle 0 reads:

```json
{"i_ptm": 0.1136, "pae": 14.2007, "plddt": 0.7598, "ptm": 0.4404,
 "recycles": 0, "id": "complex.1.pdb"}
```

pLDDT and pTM are on a 0-1 scale. The same parameters and the same ColabDesign
version give these numbers again; small differences in the last decimals across
GPU models or CUDA versions are normal.

## Generating templates for a new target

1. Predict the complex with LocalColabFold using MSAs, and take the monomers of
   the top-ranked model. Discard their relative arrangement.
2. Dock the two monomers with ZDOCK 3.0.2 at default settings, which yields
   2000 ranked poses, and keep the first 100.
3. For a poly-X target sequence, rebuild the side chains of each pose onto that
   sequence with SCWRL4, leaving the backbone unchanged.

Residue insertion codes must be resolved and residues renumbered from one, so
that numbering is consistent across all the tools involved.
