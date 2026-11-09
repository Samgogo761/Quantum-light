# ZnO wurtzite QE + Wannier90 input suite

Generated for a quick ZnO validation of the SBE pipeline.

- Structure: wurtzite ZnO, 4 atoms, no SOC, nonmagnetic PBE.
- QE cell: `ibrav=4`, `celldm(1)=6.138000` bohr, `celldm(3)=1.602000`.
- Wannier cell in Angstrom: a=3.24808972 A, c=5.20343973 A.
- Pseudopotentials expected under `./pseudo/`.
- NSCF and Wannier90 k-points are the same explicit 8x8x6 Gamma-centered mesh.
- `ZnO.win` deliberately leaves energy windows unset until `ZnO.eig` and the first interpolation are inspected.

Server upload sketch:

```bash
cd /public/home/wangjs/project
mkdir -p ZnO_TB/pseudo
# upload/copy this directory content into ZnO_TB/
# copy pseudo files into ZnO_TB/pseudo/
cd ZnO_TB
bash run_all.sh
```

If modules require MPI wrappers, override commands, e.g.:

```bash
PW_COMMAND="mpirun -np 36 pw.x" PW2WAN_COMMAND="pw2wannier90.x" W90_COMMAND="wannier90.x" bash run_all.sh
```
