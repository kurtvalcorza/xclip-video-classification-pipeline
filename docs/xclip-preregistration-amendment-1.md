# Amendment 1 to the X-CLIP pre-registration (2026-10-18)

Amends `docs/xclip-preregistration.md` (commit `134af58`). Committed alone, before any score on the fresh test. No
one has scored the fresh test: CPU checks ran with it masked, and no hosted run has happened. The trigger is the
independent peer review of PR #9 (findings F1, F5 and F6). It was a name analysis of Appendix A; nothing was
downloaded, decoded or scored. The amendment changes the film key, the fresh-test list derived from it, the bootstrap
clusters, and the reporting around the reading rule. Everything else in the pre-registration stands: the grid, the
fixed settings, the selection rule, the keep-frozen margin, the score-once rule and the verdict rule.

## A1. Normalised film key (replaces the "strict" key of §4)

The film key is the source-video name from the file name `<film>_<class>_<tags>_<n>.avi`, transformed in this order:

1. case-fold;
2. delete every character that is not `0-9` or `a-z`;
3. strip trailing digits.

`samples.film_of` implements this. The strict key of §4 kept underscores and punctuation, so spelling variants of one
film escaped it. The 300 sample clips include `treppensteigen_1`, `Basketball_Dribbling_Tips__2`, `hecht_sprung` and
`Zwei_hoffnungslose_Pflegef_lle_beim_Turnen(Part2)`. The fresh pool includes `Treppen_steigen`,
`Basketball_Dribbling_Tips`, `Hechtsprung_1` and `…(Part4)`. The rule is applied mechanically and nothing is
judged case by case. Near-names that differ after normalisation stay in the pool (for example `fun_chewing_gum!` and
`chewing_bubble_gum` against `chewing_gum`). Film identity beyond the name is not screened, and neither is actor
identity across different films; both are residual exposure, like the Kinetics-400 overlap.

## A2. Re-derived fresh test

The pool is the same: `default/train/0000.parquet` row groups 0–2, with the same row-group pins. A clip is excluded
when its normalised film key equals the normalised key of any of the 300 sample films. There are 10 keys:

- `basketballdribblingtips`
- `chewinggum`
- `findingforrester`
- `hechtsprung`
- `howtodoacartwheel`
- `projektspoho2008270208trainingbodenkr`
- `returnoftheking`
- `theboondocksaints`
- `treppensteigen`
- `zweihoffnungslosepflegefllebeimturnenpart`

The near-duplicate screen of §4 removed none and stays as it was.

Clips removed by this amendment, in addition to the 18 removed under §4 (row group, index in row group, path):

| rg | index | path |
|---|---|---|
| 0 | 134 | `Zwei_hoffnungslose_Pflegef_lle_beim_Turnen(Part4)_cartwheel_f_cm_np1_ri_bad_0.avi` |
| 1 | 172–180 | `Treppen_steigen_climb_stairs_f_cm_np1_{ba_bad_3,ba_bad_4,ba_bad_5,le_bad_6,le_bad_7,le_bad_8,ri_bad_0,ri_bad_1,ri_bad_2}.avi` (9 clips) |
| 1 | 229 | `Hechtsprung_1_dive_f_cm_np1_ba_bad_0.avi` |
| 2 | 59 | `Basketball_Dribbling_Tips_dribble_f_cm_np1_ri_med_4.avi` |
| 2 | 60 | `Basketball_Dribbling_Tips_dribble_f_cm_np1_ri_med_6.avi` |

Result:

- **669 clips** in 242 films (largest film 12 clips; Σm²/n = 4.01).
- Per class: brush_hair 70, cartwheel 66, catch 68, chew 68, clap 70, climb 69, climb_stairs 57, dive 67, draw_sword 68, dribble 66. Every class is at or above 50.
- The list is Appendix A of the pre-registration minus the 13 rows above.
- Its `fresh_clip_list_digest` is **`c651eb98607c6fca4ae733df01fa7c039df4ea18d25b6499cd3cf5ad8261abba`**, which replaces `ac0144fa…bc04`. The new value goes in `FRESH_TEST_CLIPS = 669` and `FRESH_TEST_LIST_SHA256`.

## A3. Bootstrap clusters

The film-cluster bootstrap of §6 resamples films under the A1 key. Spelling variants of one series are now one
cluster: for example `boom_snap_clap`, `boom-snap-clap` and `boomsnapclap!` all become `boomsnapclap`.

## A4. Minimum detectable effect (stated before the score)

This is the 80 % power, two-sided 0.05 minimum detectable paired top-1 difference on the 669 clips, using a
normal approximation. The formula is MDE = 2.80 × √(d / n) × √(1 + (Σm²/n − 1) × ICC), where d is the discordant
fraction and ICC is the within-film correlation. The inputs are assumed; none are measured on the fresh test.

| d \ ICC | 0.0 | 0.1 | 0.3 |
|---|---|---|---|
| 0.10 | 0.034 | 0.039 | 0.047 |
| 0.14 | 0.041 | 0.046 | 0.056 |
| 0.20 | 0.048 | 0.055 | 0.067 |

A true gain below about 0.04–0.06 will often read "no demonstrated gain". That reading means "not detectable at this
size", not "no gain". The notebook says so beside the verdict.

## A5. McNemar: clustered and descriptive only

The report gives:

- the discordant counts;
- the exact two-sided McNemar p, which assumes independent clips;
- the cluster-adjusted McNemar statistic of Durkalski et al. (2003): X² = (Σ_k (b_k − c_k))² / Σ_k (b_k − c_k)² over films k, with 1 degree of freedom.

Both p values are **descriptive**. The verdict is decided by the bootstrap interval alone (§6, item 4, unchanged).

## A6. Recording the device

Every fresh-test look records the device and the GPU name in its report and in the look ledger. This makes a non-T4
look identifiable. The definition of the evaluation of record (§5) is not changed by this amendment.
