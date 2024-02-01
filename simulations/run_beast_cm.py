#!/usr/bin/env python
"""
Script to run the BEAST on the Scylla-like data.
"""

# system imports
import argparse

# BEAST imports
from beast.tools.run import (
    create_physicsmodel,
    make_ast_inputs,
    create_obsmodel,
    run_fitting,
)
from beast.physicsmodel.grid import SEDGrid
from beast.observationmodel.observations import Observations
import beast.observationmodel.noisemodel.generic_noisemodel as noisemodel
from beast.fitting import trim_grid
from beast.tools import beast_settings, simulate_obs


def simulate_main_grids(settings):

    nsim = 10000

    print("simulating grid")
    print(settings.filters)
    print(settings.basefilters)
    print(settings.project)

    create_physicsmodel.create_physicsmodel(
        settings,
        nsubs=settings.n_subgrid,
        nprocs=1,
    )
    create_obsmodel.create_obsmodel(
        settings,
        use_sd=False,
        nsubs=settings.n_subgrid,
        nprocs=1,
    )

    physicsgrid = "{0}/{0}_seds.grid.hd5".format(settings.project)
    obsgrid = "{0}/{0}_noisemodel.grid.hd5".format(settings.project)
    output_catalog = "{0}.fits".format(settings.project)

    simulate_obs.simulate_obs(
        physicsgrid,
        obsgrid,
        output_catalog,
        nsim=nsim,
        compl_filter="F475W",
        magcut=30,
        weight_to_use="uniform",
    )


def fit_sims(settings):

    settings.obsfile = "{0}.fits".format(settings.project)
    print(settings.obsfile)

    print("Trimming the model and noise grids")

    # read in the observed data
    obsdata = Observations(settings.obsfile, settings.filters, settings.obs_colnames)

    # get the modesedgrid on which to generate the noisemodel
    modelsedgridfile = settings.project + "/" + settings.project + "_seds.grid.hd5"
    modelsedgrid = SEDGrid(modelsedgridfile)

    # read in the noise model just created
    noisemodel_vals = noisemodel.get_noisemodelcat(settings.noisefile)

    # trim the model sedgrid
    sed_trimname = "{0}/{0}_seds_trim.grid.hd5".format(settings.project)
    noisemodel_trimname = "{0}/{0}_noisemodel_trim.grid.hd5".format(settings.project)

    trim_grid.trim_models(
        modelsedgrid,
        noisemodel_vals,
        obsdata,
        sed_trimname,
        noisemodel_trimname,
        sigma_fac=3.0,
    )

    run_fitting.run_fitting(
        settings,
        use_sd=False,
        nsubs=settings.n_subgrid,
        nprocs=1,
        pdf2d_param_list=[],  # "Av", "M_ini", "logT"],
    )
