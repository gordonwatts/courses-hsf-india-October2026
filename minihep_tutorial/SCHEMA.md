# Workshop ROOT schema and units

## Reduced ATLAS Open Data workshop ROOT file

`dataWW_d1.root` contains a `tree_event` TTree with 600,000 entries. Its 30 branches are flat scalar columns. The tutorial selects `lep_n == 2`, leaving 596,571 eligible entries before CPU-size sampling. The source is the [course repository](https://github.com/gordonwatts/courses-hsf-india-October2026/blob/main/dataWW_d1.root); the existing [loader](https://github.com/gordonwatts/courses-hsf-india-October2026/blob/main/extra_functions.py) also selects two-lepton events but additionally requires positive `mcWeight` for its supervised exercise.

| Branches | Meaning and units | MiniHEP use |
| --- | --- | --- |
| `lep_n` | reconstructed lepton count | select exactly 2 |
| `lep_pt_0/1`, `lep_eta_0/1`, `lep_phi_0/1` | GeV, unitless, radians | lepton momentum components |
| `lep_E_0/1` | **MeV** | divide by 1,000 to get GeV energy |
| `lep_type_0/1`, `lep_charge_0/1` | absolute PDG flavor 11/13 and charge ±1 | discrete e±/μ± token types |
| `jet_n` | total jet count, possibly greater than 2 | presence of stored jet slots |
| `jet_pt_0/1`, `jet_eta_0/1`, `jet_phi_0/1` | GeV, unitless, radians | up to two jet momentum components |
| `jet_E_0/1` | **MeV** | divide by 1,000 to get GeV energy |
| `met_et`, `met_phi` | GeV, radians | transverse proxy token `(MET cosφ, MET sinφ, 0, MET)` |
| `channelNumber`, `eventNumber` | event identifiers | split integrity checks only |
| `label` | signal/background indicator | returned for later diagnostics; never a model input |
| `mcWeight` | simulation event weight, can be negative | unused in this introductory pretraining loss |
| `runNumber`, `index` | metadata | never a model input |

When jet_n is 0 or 1, unused jet columns contain -7 sentinels. The loader uses jet_n to identify those absent jets, creates zero padded slots, and sets the padding mask.
