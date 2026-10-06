# Coefficienti di ricalibrazione

`recalibration_final.csv` contiene le rette `CL = a·SUVr + b` stimate nella tesi
per ciascun tracciante (FBB, FMM, FBP) e ciascun metodo, sia per regressione
(la calibrazione adottata) sia con il two-point, sui soggetti GAAIN allineati
con ANTs. È una copia del file prodotto da
`scripts/analisi/build_final_comparison.py` in
`data/comparison_all_methods/`.

Serve all'interfaccia e a `scripts/run_pipeline.py` per mostrare il centiloide
ricalibrato anche a chi non ha la cartella `data/`. Se il file in `data/` esiste
(per esempio dopo aver rifatto le analisi), il codice usa quello.

Non contiene dati di pazienti, solo coefficienti.
