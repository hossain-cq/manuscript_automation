# Golden project

The chosen golden/first-domain project is **not** copied into this repo — it's
a real, already-published project:

```
/home/user/Hossain/AQT_electrolyte
```

*Quantum Simulation of Battery Electrolyte Salts*, published in *Advanced
Quantum Technologies* (DOI 10.1002/qute.202500871). Being already published
makes it a good golden project: there's a ground-truth paper to check the
system's reconstruction against, and repeated read-only scans carry zero risk
to active research.

Run the CLI against it directly:

```bash
python -m manuscript_system.cli intake --path /home/user/Hossain/AQT_electrolyte --domain quantum_chemistry
```

Then verify the read-only boundary held:

```bash
cd /home/user/Hossain/AQT_electrolyte && git status
# must be identical before and after the intake run above
```

`configs/domains/quantum_chemistry.yaml` was seeded from this project's own
`README.md` (method vocabulary, expected artifact types, completeness
checklist) — see that file for what's currently checked.
