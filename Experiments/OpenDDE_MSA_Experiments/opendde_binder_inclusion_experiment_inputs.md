# OpenDDE Binder-Inclusion Experiment — Inputs

## Summary & Goals

Testing whether including the binder chain in OpenDDE's paired MSA search (vs. excluding it — target-only) meaningfully affects refiner outcomes. This follows from two earlier findings that pointed in different directions:

- **Quantitative** (single original example): excluding the binder from the paired search gave a *lower* iptm on that one design — suggesting inclusion helps.
- **Qualitative** (same example, visual check): the `GFPGER` motif — expected to sit near the ligand — was correctly placed in the *included* structure but not the *excluded* one, even though `excluded` scored higher on a later replicate of that same design. This is a reminder that iptm is a confidence score, not a correctness score, and the two don't always agree.

This batch of 7 additional binder/target pairs (rows 3-9, `opendde_binder_inclusion_trials.csv`) is a replicate confirmatory experiment: 1 design × 7 cycles, run twice per row (binder included vs. excluded from the paired search), to see whether a consistent quantitative pattern holds across different binders — some containing the `GFPGER` motif, some not — before drawing conclusions from iptm alone.

**Scoping note:** every binder tested here is a de novo scaffold with no real evolutionary homologs, tested against a natural target with real UniRef depth. The `excluded` win is most plausibly explained by that asymmetry (see `opendde_msa_findings_2026-09-14.md` sections 13-14) — `included` jointly pairs a de novo binder that has no genuine coevolutionary history against the target's real homologs, risking noise (spurious cross-species matches) rather than adding real signal; `excluded`'s solo paired-search call was confirmed to trigger no real search at all (it just echoes the query back), so it's better described as opting out of that noise than as getting a "clean" search. This conclusion is specific to the de novo-binder case and should be re-tested before being applied to a natural (non-de-novo) binder, where real cross-chain coevolution could exist.

**Shared across every row:**
- `seq_target`: `LIDVVVVCDESNSIYPWDAVKNFLEKFVQGLDIGPTKTQVGLIQYANNPRVVFNLNTYKTKEEMIVATSQTSQYGGDLTNTFGAIQYARKYAYSAASGGRRSATKVMVVVTDGESHDGSMLKAVIDQCNHDNILRFGIAVLGYLNRNALDTKNLIKEIKAIASIPTERYFFNVSDEAALLEKAGTLGEQIFSI`
- `ligand`: `[Mg+2]`
- `hotspots` (epitope residues): `B9,B10,B11,B12,B13,B14,B46,B47,B48,B70,B72,B73,B74,B75,B76,B77,B78,B79,B112,B113,B114,B116,B143`
- Model: OpenDDE, `--num-cycles 7 --num-designs 1 --num-samples 5`

**What varies per row:** `seq_binder`, `fixed_residues`, and the `msa_options` value that controls the experiment condition.

## Inputs

| Row | seq_binder | fixed_residues | msa_options (binder included) | msa_options (binder excluded) |
|---|---|---|---|---|
| 3 | `MNPLLEANKESLKKEVEYLSALIAAYESVLASLGVEVEKVDDPSARPGERVPGTGIRSVIKDGKTYRFVVHPDGTAELHPDETDKEAAAEAIKAAIYNYLKPSKEALEKELEKL` | `A48 A49 A50` | `,` | `empty,` |
| 4 | `VVKSAEEAIEYIEKNKDKEEIVVTIDMEDSVEGLKAALEVTKYAIEKGIKTPITIRLGERKSKELPPLEEAKKIDKENFELVEELAELVAQAPNISVRYQAVNYTAMLYMKEGKLDEAKKLTEETLALTERLQ` | `A58 A59 A60` | `,` | `empty,` |
| 5 | `MEEGRERGLRVAAEIGKNKDNLEKVAEIYREALLELGVSEEHAELSAERVKAGGFPGERRATSEADLVAESIVNGAVNAIVLGVEKTIEAYKEGLESAKKAGLPEEAIRNSEAALEGAEVVKEAA` | `A54 A55 A56 A57 A58 A59` | `,` | `empty,` |
| 6 | `MLVAPTGLSPREQELADFLQEKTKEIRAAFPDGSLRIGERERVGDETIVTLTYKEGDEKAKAAAEKVAKEIREKFPDLVVRIEA` | `A38 A39 A40` | `,` | `empty,` |
| 7 | `SKPATREDLEKAKKIQEELGVESMSDETIESMLGVVTEEDVSSQIVTSYVAPILAENEGKIGFPGERQPNREAALEEARRQIERAQKLGQDIVLIRVSPEQADLEEELRALVAEYPDLKGIIIPPTTLENVKEVAEETKKALEKA` | `A62 A63 A64 A65 A66 A67` | `,` | `empty,` |
| 8 | `MVKEAIRKVVERKVPLEDVKDLIAKASREEVAKTILEAIKAGAGFPGERTPDTPKKIDEAMIEAYSSLTEKDPHYLGALENQAALLARLGETEKARELFRKLDKLSGSELPEEFLEELIERIK` | `A44 A45 A46 A47 A48 A49` | `,` | `empty,` |
| 9 | `LDAEAVLRELLEAAERGDLERARRISLEAFEKLKDKLPEETYEESVKGLEEGGFPGERPKPISSVKDPEEKTALQLLGAASLAVGAAVAAQEGKREEAVERLREELEELL` | `A53 A54 A55 A56 A57 A58` | `,` | `empty,` |

Note: `msa_options=","` splits to `['', '']` (both binder and target searched, paired together). `msa_options="empty,"` splits to `['empty', '']` (binder excluded, target searched alone). Design names follow `opendde_row{N}_binder_{included,excluded}`.

## Results

`best_cycle`/`best_iptm` from each run's `refined_designs.csv`. Motif-ligand distance is the minimum atom-atom distance between the chain-A residues listed in `fixed_residues` (used here as a proxy for the binder's functional motif) and chain C (the ligand), computed directly from the downloaded PDB via biotite. "In contact" uses the same ≤4.5Å cutoff as `StrucTools.determine_binding_interface`. Confirmed the `fixed_residues` window exactly matches the `GFPGER` motif's sequence position for rows 5, 7, 8, 9; rows 3, 4, 6 intentionally fix a shorter 3-residue `GER` motif specific to those scaffolds, not the full 6-residue `GFPGER` — so those three rows are testing a shorter/less stringent window than the other four.

| Row | Included cycle | Included iptm | Included motif→ligand dist | Excluded cycle | Excluded iptm | Excluded motif→ligand dist | Motif in contact (≤4.5Å) |
|---|---|---|---|---|---|---|---|
| 3 | 7 | 0.336 | 17.00 Å | 2 | **0.826** | **2.16 Å** | excluded only |
| 4 | 4 | 0.628 | 29.27 Å | 7 | **0.955** | **4.25 Å** | excluded only |
| 5 | 7 | 0.696 | 7.87 Å | 1 | **0.840** | **1.91 Å** | excluded only |
| 6 | 2 | 0.567 | 3.66 Å | 4 | **0.913** | 2.29 Å | both |
| 7 | 4 | 0.356 | 15.00 Å | 6 | 0.828 | 19.73 Å | neither |
| 8 | 1 | 0.802 | 9.50 Å | 7 | **0.926** | **1.90 Å** | excluded only |
| 9 | 3 | 0.447 | 18.55 Å | 3 | **0.945** | **1.85 Å** | excluded only |

**Both** iptm and motif-ligand contact favor `excluded` in 5/7 rows, tie in 1/7 (row 6 — both in contact), and fail in 1/7 (row 7 — neither in contact, despite `excluded` scoring 0.828 iptm there, underscoring that iptm doesn't reliably track this correctness check even within the winning condition). `included` never uniquely achieves motif-ligand contact in any row. This reverses the lean toward `included` drawn from the single original example (see `opendde_msa_findings_2026-09-14.md` section 11) — with 7 replicates and a direct quantitative check instead of a visual read on one structure, `excluded` (target-only paired search) now looks better on both criteria. The original example may simply have been an outlier.

### Cycle 0 (same starting sequence, condition-only comparison)

Cycle 0 predicts the structure for the *original, un-redesigned* binder sequence — identical in both conditions for a given row, so any difference here is caused purely by the MSA condition (paired search on vs. off for the binder), with no confound from MPNN having picked a different redesigned sequence per condition (unlike the best-cycle table above, where `included` and `excluded` end up comparing two different MPNN-generated sequences).

| Row | Cycle 0 iptm (included) | Cycle 0 iptm (excluded) | Cycle 0 motif dist (included) | Cycle 0 motif dist (excluded) | Contact? |
|---|---|---|---|---|---|
| 3 | 0.332 | **0.653** | 6.05 Å | **1.94 Å** | excluded only |
| 4 | 0.447 | **0.843** | 17.75 Å | **3.37 Å** | excluded only |
| 5 | 0.393 | **0.826** | 7.53 Å | **1.47 Å** | excluded only |
| 6 | 0.215 | **0.910** | 6.73 Å | **1.59 Å** | excluded only |
| 7 | 0.228 | 0.295 | 16.35 Å | **2.11 Å** | excluded only |
| 8 | **0.766** | 0.385 | 8.85 Å | **3.71 Å** | excluded only |
| 9 | 0.261 | **0.838** | 3.23 Å | 1.89 Å | both |

At cycle 0, `excluded` achieves motif-ligand contact in **all 7 rows** (6 uniquely, 1 shared with `included`) — a cleaner and stronger result than the best-cycle table, since the sequence itself is held constant. `included` only achieves contact once (row 9, shared). Row 8 is the one case where `included`'s iptm is higher at cycle 0, yet `excluded` still gets the motif correctly placed — a second, independent instance of iptm and motif-correctness disagreeing. This cycle-0-only comparison is the most controlled evidence so far in favor of `excluded` as the better default.
