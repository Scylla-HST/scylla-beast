import numpy as np

from astropy import units

import argparse

from beast.tools import beast_settings
from beast.observationmodel.noisemodel import absflux_covmat
import sys

sys.path.insert(0, "/ocean/projects/ast190023p/cmurray3/new_sim/")

import run_beast_cm

if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--physicsmodel",
        help="Generate the main simulated physics model grid and noisemodel grid",
        action="store_true",
    )
    parser.add_argument(
        "-f", "--fit", help="Fit the simulated data", action="store_true"
    )
    args = parser.parse_args()

    # read in BEAST settings
    settings = beast_settings.beast_settings("beast_settings_highavdist_sim.txt")

    settings.obs_colnames = [f.upper() + "_RATE" for f in settings.basefilters]
    settings.ast_colnames = np.array(settings.basefilters)
    settings.noisefile = (
        settings.project + "/" + settings.project + "_noisemodel.grid.hd5"
    )

    settings.absflux_a_matrix = absflux_covmat.hst_frac_matrix(settings.filters)
    settings.add_spectral_properties_kwargs = dict(filternames=settings.filters)

    if args.physicsmodel:
        run_beast_cm.simulate_main_grids(settings)

    if args.fit:
        run_beast_cm.fit_sims(settings)
