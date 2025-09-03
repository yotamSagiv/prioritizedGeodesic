# Between planning and map-building: prioritizing replay when future goals are uncertain

This repo contains the modeling and plotting code from "Between planning and map-building: prioritizing replay when future goals are uncertain" (Sagiv et al. 2025). Namely:

* Figure*.ipynb contains the plotting code that loads in the simulated data for Figure(s) * and generates the corresponding figure panels.
* figs/ contains the raw Adobe Illustrator files for the main and supplemental figures, plus the generated PDFs
* Data/*/ contains the simulated data for *
* gillespie_GR.py/gillespie_mattar.py contain the GR/Q-learning models of the Gillespie task (Figure 3)
* carey_GR.py/carey_mattar.py contain the GR/Q-learning models of the Carey task (Figure 4)
* pfeiffer.py contains the GR model of the Pfeiffer task (Figure 5)
* prediction.py contains the GR model of the prediction task (Figure 6)
* geodesic_agent.py contains all the code necessary for running a GR agent (particularly of note are the `replay()`, `dynamic_replay()`, and `foster_replay()` methods, which implement prioritized replay in different settings).
* reward_agent.py is the analogous Q-learning agent file.
* MarkovDecisionProcess.py, gridworld.py, and graph.py contain infrastructural RL code (e.g., an MDP class, a GridWorld class, etc.)
* All other .py files contain utility functions that accomplish various useful, but technically independent, aims (plotting, computing occupancy matrices, etc.)

### Author

Yotam Sagiv
