# CICT 2026 (Paper 758) – camera-ready: what changed, what you must check

## Files
- CICT758_overleaf.zip  – Overleaf project (upload, main file main.tex, compile with pdfLaTeX). Based on your SUBMITTED paper, not the repo's main.tex.
- CICT758_supplement.zip – supplementary_tables.pdf (+ .tex, CSVs, script to regenerate)
- CICT758.docx – editable content copy (paste into the IEEE A4 Word template; check equations and Table III)

## Must do before upload (in this order)
1. Replace the two red affiliation placeholders (\affA, \affB near the top of main.tex).
2. Confirm creditcard feature count d (Table I says 29; the data loader keeps Time and Amount, which gives 30). Run:
   python -c "from src.data_loader import load_dataset; print(load_dataset('creditcard')[0].shape)"
   Fix Table I (search for "CHECK d for creditcard" in main.tex) if needed.
3. Compile in Overleaf; page count must be <= 6. If it spills over, cut in this order: Section IV-F "Cost-sensitive learning" paragraph, the sweeps sentence in IV-E, limitation (7).
4. Confirm these statements are true, or edit them: (a) defaults k=5, alpha=2, beta=0.7 were not tuned on the results (III-B); (b) no dataset was dropped after seeing results (III-C); (c) ablation datasets were picked by judgment (IV-E).
5. Re-run Turnitin (< 20 %), then PDF eXpress (Conference ID 72580X).

## Reviewer comments: how each was handled
Done in text/analysis (from your existing data):
- "indistinguishable" -> "no evidence of benefit"; equivalence (TOST), bootstrap CIs, minimum detectable difference added (IV-D, Table IV, supplement)
- exploratory vs main analysis separated (III-D); family definition for every test; joint adjustment over 140 tests reported
- plain-SMOTE row added to ablation (Table V; taken from main grid, same folds); n=5 power floor stated
- five seed-check methods named; defaults and dataset selection explained; code link added
- timing section corrected (see below); limitations expanded (unweighted surrogate, SHAP on minority only, min-max normalisation, FW-SMOTE/K-means not compared, asymmetric tuning, no threshold tuning/balanced ensembles)
- full pairwise tables (all 5 metrics, raw/Holm/BH p, rank-biserial) in the supplement
- authors/affiliations/emails, no we/our/us, required section names, \gmean line-break fix
NOT done (needs new experiments; stated as limitations/future work): standardized RF/XGB multi-seed, FW-SMOTE and K-means SMOTE baselines, threshold tuning, balanced ensembles, attribution-stability and neighbour-overlap diagnostics, isolated re-timing, more datasets.

## Problems found in the package you sent
- 2_STRONGLY_RECOMMENDED/code/overleaf_project/main.tex is a DIFFERENT, older draft (other title, 18 refs, rewritten abstract, wrong "Holm-Bonferroni", no Problem Statement). Do not use it; replace it in GitHub with the new main.tex.
- Released experiment.py times the resampler only, but the timings in raw_results.csv (6 methods) include classifier training. Either re-run timing with the current code or state the mismatch in the README.
- answers.txt: the creditcard timing explanation ("SMOTE is single-threaded Python loops") is not supported by the code (AC-SMOTE is the Python loop); the "defaults chosen because..." text is post-hoc. Neither was used in the paper.
- Datasets are from Kaggle (loader/manifest), not OpenML as the submitted paper said; corrected.
- Literature check: web search only (no IEEE Xplore/Scopus). Repeat it there before claiming novelty more strongly. Possibly relevant leads not cited (could not verify authors): K-IPO (arXiv 2607.16478), a SHAP-conditioned diffusion oversampler thesis (METU).
- New references 22-23 (Schuirmann 1987; Lakens 2017) are from memory: please verify volume/pages before submission.
