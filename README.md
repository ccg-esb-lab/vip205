# Conditional filamentation enhances bacterial survival in toxic environments

Code used to generate figures and analyses for:

**Conditional filamentation enhances bacterial survival in toxic environments**  
O.B. Aguilar-Luviano, F. Santos-Escobar, S. Orozco-Barrera, and R. Peña-Miller

**Data:** https://doi.org/10.5281/zenodo.19239573

## Overview

This repository contains all the dta analysis and simulation code used to explore how inducible filamentation in *Escherichia coli* modulates survival under stress. We combined experimental assays (microchemostat, flow cytometry, and mother machine) with a simple mathematical model to quantify toxin accumulation and investigate the protective role of filamentation under exposure to antibiotics and heavy metals.

---

## Figures

### Figure 1

- [`VIP205_Fig1B_Filamentation.Rmd`](./VIP205_Fig1B_Filamentation.Rmd)  
  Flow cytometry analysis of filamentation across stress conditions. Includes gating, SSC-based cell size distributions, and quantification of the fraction of elongated cells.

- [`VIP205_Fig1D-H_Model.ipynb`](./VIP205_Fig1D-H_Model.ipynb)  
  Mathematical model and simulations of toxin accumulation during filamentation. Includes surface area-to-volume scaling, intracellular toxin dynamics, heterogeneity, and survival predictions.

---

### Figure 2

- [`VIP205_Fig2_Microchemostat.ipynb`](./VIP205_Fig2_Microchemostat.ipynb)  
  Microchemostat time-lapse analysis. Includes image loading, cell geometry measurements, single-cell trajectories, and quantification of surface area-to-volume changes during elongation.

---

### Figure 3

- [`VIP205_Fig3_DilutionCytometry.Rmd`](./VIP205_Fig3_DilutionCytometry.Rmd)  
  Flow cytometry analysis of intracellular dye dilution. Includes time-resolved fluorescence measurements and comparison between dividing and elongating cells.

- [`VIP205_Fig3_DilutionMicroscopy.Rmd`](./VIP205_Fig3_DilutionMicroscopy.Rmd)  
  Microscopy-based quantification of intracellular dilution. Includes single-cell tracking, cell length dynamics, and fluorescence dilution over time.

- [`VIP205_Fig3_Dilution_deltaRecA.Rmd`](./VIP205_Fig3_Dilution_deltaRecA.Rmd)  
  Equivalent dilution analysis in the ΔrecA background.

---

### Figure 4

- [`VIP205_Fig4_MotherMachine_analysis.ipynb`](./VIP205_Fig4_MotherMachine_analysis.ipynb)  
  Baseline mother machine analysis before stress exposure. Includes elongation rates, division timing, and growth dynamics under control conditions.

- [`VIP205_Fig4_MotherMachineAMP.ipynb`](./VIP205_Fig4_MotherMachineAMP.ipynb)  
  Mother machine analysis under ampicillin exposure. Includes single-cell trajectories, survival classification, and length-dependent survival.

- [`VIP205_Fig4_MotherMachineCd.ipynb`](./VIP205_Fig4_MotherMachineCd.ipynb)  
  Mother machine analysis under cadmium exposure. Includes growth, elongation, and survival as a function of cell length.

---

### Figure 5

- [`VIP205_Fig5_DeltaRecA_cytometry.ipynb`](./VIP205_Fig5_DeltaRecA_cytometry.ipynb)  
  Flow cytometry analysis of survival under toxic stress. Includes PI-based viability measurements, time-resolved survival curves, and comparisons between elongated and non-elongated cells.

---

## Image Analysis Code

**[py_MotherMachine.py](py_MotherMachine.py)**  
Python script for processing kymograph data from mother machine experiments. Extracts single-cell trajectories to quantify elongation rates, division timing, and survival outcomes under different stress conditions.

**[muPy_viewer.py](muPy_viewer.py)**  
Interactive Python tool for manual correction of segmentation and tracking errors in microchemostat time-lapse experiments. Supports frame-by-frame validation of cell trajectories and fluorescence intensity measurements.

---

## Authors

[OBAL/RPM] [@Systems Biology Lab, CCG-UNAM](https://github.com/ccg-esb-lab)

---

## License

This project is licensed under the [MIT License](./LICENSE).
