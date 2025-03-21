# Conditional filamentation enhances bacterial survival in toxic environments

Scripts and data used to produce figures and simulations related to the article:  
**Conditional filamentation enhances bacterial survival in toxic environments**  
*O.B. Aguilar-Luviano, F. Santos-Escobar, S. Orozco-Barrera, and R. Peña-Miller*

---

## Overview

This repository contains all the analysis and simulation code used to explore how inducible filamentation in *Escherichia coli* modulates survival under stress. We combined experimental assays (microchemostat, flow cytometry, and mother machine) with a simple mathematical model to quantify toxin accumulation and investigate the protective role of filamentation under exposure to antibiotics and heavy metals.

---

## Image Analysis Code

**[py_MotherMachine.py](py_MotherMachine.py)**  
Python script for processing kymograph data from mother machine experiments. Extracts single-cell trajectories to quantify elongation rates, division timing, and survival outcomes under different stress conditions.

**[muPy_viewer.py](muPy_viewer.py)**  
Interactive Python tool for manual correction of segmentation and tracking errors in microchemostat time-lapse experiments. Supports frame-by-frame validation of cell trajectories and fluorescence intensity measurements.

## Data Analysis Notebooks

**[VIP205_Model.ipynb](VIP205_Model.ipynb)**  
Implementation of a simple ODE model capturing toxin uptake and dilution during filamentation. Includes analytical exploration of SA/V dynamics and simulation of accumulation profiles under different growth modes.


**[VIP205_MotherMachine_analysis.ipynb](VIP205_MotherMachine_analysis.ipynb)**  
Preprocessing and general exploration of mother machine data, including IPTG induction effects and baseline growth parameters.

**[VIP205_Microchomostat_analysis.ipynb](VIP205_Microchomostat_analysis.ipynb)**  
Analysis of microchemostat experiments tracking OD dynamics under AMP and Cd exposure. Includes calculation of maximum growth rates and comparisons between filamented and control strains.

**[VIP205_MotherMachine_analysis_Cd.ipynb](VIP205_MotherMachine_analysis_Cd.ipynb)**  
Single-cell tracking in the mother machine under cadmium exposure. Quantifies elongation, survival, and division dynamics of filamented and control cells, and analyzes the impact of cell length on survival outcomes.

**[VIP205_MotherMachine_analysis_AMP.ipynb](VIP205_MotherMachine_analysis_AMP.ipynb)**  
Same as above, but for ampicillin exposure. Includes filamentation tracking, cell division analysis, and survival classification from time-lapse data.

**[VIP205_MotherMachine_analysis.ipynb](VIP205_MotherMachine_analysis.ipynb)**  
Comparison of baseline growth dynamics between filamented and control strains prior to stress exposure in the mother machine. Includes IPTG classification, elongation and division rate measurements, and analysis of strain-specific differences.

**[VIP205_cytometry_analysis.ipynb](VIP205_cytometry_analysis.ipynb)**  
Analysis of flow cytometry data to quantify cell size and survival after exposure to toxic agents using propidium iodide staining. Identifies morphological correlates of survival based on forward scatter measurements.

## Authors

[@Systems Biology Lab, CCG-UNAM](https://github.com/ccg-esb-lab)

## License

This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License - see the [license](LICENSE) file for details. 


