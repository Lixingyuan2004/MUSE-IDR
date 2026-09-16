# C1 manuscript materials

This stage converts locked B6 confirmatory statistics and B7 descriptive error
analysis into manuscript-ready tables, figures, and prose drafts. It does not
read raw CAID labels or predictions and does not tune A10.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe experiments\analysis\c1_manuscript_materials\build_c1.py
node experiments\analysis\c1_manuscript_materials\render_svgs.js outputs\paper\c1_manuscript_materials
```

SVG files are the publication masters. PNG files are convenience previews.
