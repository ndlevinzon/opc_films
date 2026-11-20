# Simulation inputs

All production runs are NVT. Temperature along a cooling window is lowered with Amber time-dependent restraints (`nmropt`). The time step is 2 fs with SHAKE on bonds to hydrogen, and the Langevin collision frequency is 2 ps^-1. Coordinates during cooling are written every 2 ps (`ntwx = 1000`). Holds are written every 10 ps (`ntwx = 5000`).

Cooling windows use `cut = 9.0`. Equilibration and minimization use `cut = 8.0`. The archived `prop_trajectory.mdin` still says `cut = 8.0`. The methods text should report 9 Å for the cooling runs.

| Script | Role |
| --- | --- |
| `minimize_equilibrate_300K.gpu` | Four minimizations and NVT equilibration at 300 K. |
| `cool_300K_to_78K_nvt.gpu` | One temperature window, then submits the next window. |
| `prop_trajectory.gpu` | Continues a configuration at fixed temperature. |
| `prop_trajectory.mdin` | Archived continuation input. `temp0` is 300 K, `nstlim` is unlimited, and `cut` is 8.0. |
| `hold_from_cooling.py` | Writes a hold at `--temperature` for `--length-ns`, with `cut = 9.0` and an explicit `nstlim`, and prints the cooling restart. |
| `next-RST.perl` | Next restart index, read from `RST/` in the working directory. |
| `resolve_inputs.sh` | Finds `build/opc_slab.prmtop` by walking upward from the working directory. |

## How to run

Set `AMBERHOME` to the Amber prefix and load the CUDA and MPI modules that match that build. Submit with this machine's account and partition:

```bash
sbatch --account=ACCOUNT --partition=PARTITION --qos=QOS minimize_equilibrate_300K.gpu
sbatch --account=ACCOUNT --partition=PARTITION --qos=QOS \
    --export=ALL,COOL_NS=50 cool_300K_to_78K_nvt.gpu
```

Holds and endpoints are run from the setpoint directory, not from a path stored in this repository:

```text
opc/<replica>/hold/<temperature>K/TRAJ/traj.N
opc/<replica>/hold/<temperature>K/OUT/mdout.N
opc/<replica>/hold/<temperature>K/RST/rst.N
opc/<replica>/endpoint/<temperature>K/TRAJ/traj.N
opc/<replica>/endpoint/<temperature>K/OUT/mdout.N
```

`N` is 1, 2, … for each continuation. `prop_trajectory.gpu` writes those names. `MDIN` is required: `hold_from_cooling.py` writes `cut = 9.0`, and the archived `prop_trajectory.mdin` (`cut = 8.0`, 300 K, unlimited `nstlim`) is not the default. The catalog treats every `traj.N` in one setpoint directory as one hold. The first segment's `INPCRD` (`-c`) is `source_restart`. Later segments restart from `RST/rst.(N-1)`.

`COOL_NS` is the total cooling time in ns: 2, 5, 10, 20, or 50. Those times give 111, 44.4, 22.2, 11.1, and 4.44 K/ns across the 222 K drop. The temperature grid is 10 K windows from 300 K to 80 K and a final 2 K window to 78 K. Each window is one SLURM job. The next window is submitted with the same account, partition, and QOS as the job that finished.

`TOPOLOGY` overrides the upward search for `build/opc_slab.prmtop`. Minimization reads `COORDINATES` (default `build/opc_slab.rst7`). Cooling reads `START_RST`. The first cooling window of every replica starts from the same `min/step9.rst7` and the replicas differ by `ig = -1`. At 300 K those trajectories decorrelate within picoseconds. A 150–250 K hold must not start from that file. `hold_from_cooling.py --temperature 250 --from-rate 4.44 --length-ns 20 --out hold_250K.mdin` names the matching cooling restart and writes `temp0`. Restart temperatures are three digits, including 80 K and the final 80 K to 78 K window. The length that was run is the manifest column `duration_ps`, read from the finished mdout `TIME(PS)` span.

Cooling inputs set neither `iwrap` nor `nscm`. The archived hold input sets `iwrap = 1` and `nscm = 1000`. The analyses unwrap or recentre, so both can be read. The generated hold uses `cut = 9.0`. The archived `prop_trajectory.mdin` still says `cut = 8.0`. No finished hold mdout is in this archive, so the cutoff of the holds that were run has not been read. If that cutoff is 8 Å, the 78 K and 150–250 K trajectories are not at the 9 Å cooling cutoff.

The cooling script previously named a TIP4P topology at a fixed relative path. It now uses the same OPC topology as the rest of this repository.

The inputs that built `build/opc_slab.prmtop` and `build/opc_slab.rst7` are not in this repository. The 139,000-water OPC slab in the 194.567 × 188.781 × 300 Å box cannot be rebuilt from this tree.

`hold_from_cooling.py` writes `hold_<T>K.json` next to the mdin. The file records `source_restart`, `source_rate_K_per_ns`, `length_ns`, and `cut_A`. The catalog reads it when the restart name does not already carry the cooling rate. The temperatures and lengths that were actually run are not stored in this archive. The protocol used here is a 150–250 K hold continued from the 4.44 K/ns ramp, for the `--length-ns` given on the command line. Which restart the 300 K endpoint started from is not recorded.
