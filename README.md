# SDPD-mf_thixo

<img width="1023" height="670" alt="image" src="https://github.com/user-attachments/assets/2231c7d1-8886-4bb9-adc5-8892ea620e60" />

Smoothed Dissipative Particle Dynamics (SDPD) model for the analysis of multiphase and thixotropic fluids implemented in LAMMPS.

# Description

Modified version of SDPD (initially implemented by Morteza Jalalvand (IASBS)) including:
  
a) Explicitly bulk viscosity as presented by Espanol and Revenga, including background pressure, and generalizing for 2 and 3 dimensions.
	
b) Update non-slip in flat wall: This update included the option for non-slip boundary condition in a flat wall as described in Bian and Ellero. The correction can be turned on and off using the variable slip[k][l] where the index correspond to type of particles k and l. Thus a fluid particle type k, can interact with non-slip bc with a wall particle type l.

c) Update MF: This updated multiphase (mf) version is based on the description presented in Lei et al. 2016 (10.1103/PhysRevE.94.023304) including a pair-force contribution between particles.

d) Transient viscosity model: This updated inclues a thixitropic (viscosity transient) model to simulate complex multiphase flows

# Installation

1. Replace the files atom.cpp, atom.h, atom_vec.cpp, atom_vec.h, fix_sph.cpp and fix_sph.h with the files from the .tar
2. Copy the files pair_sdpd_taitwater_isothermal_mf.cpp and pair_sdpd_taitwater_isothermal_mf.h in the folder DPD-SMOOTH
3. Run: make mpi

# Examples

Here you will find an example (folder named “Example”) containing the files needed to run a case of the interaction of three droplets under a continuous fluid phase. The main script is called “in.sdpd_phase.2d”.
