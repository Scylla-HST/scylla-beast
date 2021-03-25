import numpy as np
import glob
import os
import types

import argparse

from beast.tools.run import (
    create_physicsmodel,
    make_ast_inputs,
    create_filenames,
)

from beast.plotting import plot_mag_hist
from beast.tools import (
    beast_settings,
    create_background_density_map,
    split_ast_input_file,
    setup_batch_beast_fit,
)

from beast.tools.density_map import BinnedDensityMap
from beast.observationmodel.observations import Observations
from beast.physicsmodel.grid import SEDGrid
from beast.observationmodel.vega import Vega
from beast.plotting import plot_mag_hist, plot_ast_histogram

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import Angle
from astropy import units as u


def beast_ast_inputs(field_name=None, ref_image=None, filter_ids=None, galaxy=None, supp=0):
    """
    This does all of the steps for generating AST inputs and can be used
    a wrapper to automatically do most steps for multiple fields.
    * make field's beast_settings file
    * make source density map
    * make background density map
    * split catalog by source density
    * make physics model (SED grid)
    * make input list for ASTs
    * prune input ASTs

    ----
    Inputs:

    field_name (str): name of field
    ref_image (str): path to reference image
    filter_ids (list): list of indexes corresponding to the filters in the
                        observation, referenced to the master list below.
    galaxy (str): name of target galaxy (e.g., 'SMC', 'LMC')
    ----

    Places for user to manually do things:
    * editing code before use
        - here: list the catalog filter names with the corresponding BEAST names
        - here: choose settings (pixel size, filter, mag range) for the source density map
        - here: choose settings (pixel size, reference image) for the background map

    """

    # the list of fields
    field_names = [field_name]

    # the path+file for a reference image
    im_path = [ref_image]
    ref_filter = ["F475W"]

    # choose a filter to use for removing artifacts
    # (remove catalog sources with filter_FLAG > 99)
    flag_filter = ["F475W"]

    # number of fields
    n_field = len(field_names)

    # Need to know what the correspondence is between filter names in the
    # catalog and the BEAST filter names.
    #
    # These will be used to automatically determine the filters present in
    # each GST file and fill in the beast settings file.  The order doesn't
    # matter, as long as the order in one list matches the order in the other
    # list.
    #
    gst_filter_names = [
        "F225W",
        "F275W",
        "F336W",
        "F475W",
        "F814W",
        "F110W",
        "F160W",
        "F657N",
    ]
    beast_filter_names = [
        "HST_WFC3_F225W",
        "HST_WFC3_F275W",
        "HST_WFC3_F336W",
        "HST_WFC3_F475W",
        "HST_WFC3_F814W",
        "HST_WFC3_F110W",
        "HST_WFC3_F160W",
        "HST_WFC3_F657N",
    ]



    for b in range(n_field):

        print("********")
        print("field " + field_names[b])
        print("********")

        # -----------------
        # data file names
        # -----------------

        # paths for the data/AST files
        gst_file = "./data/{0}/{0}.vgst.fits".format(field_names[b])# + ".vgst.fits"
        #ast_file = "./data/{0}/{0}.vgst.fake.fits".format(field_names[b])# + ".vgst.fits"
        ast_input_file = "./{0}/{0}_inputAST.txt".format(field_names[b])
        # path for the reference image (if using for the background map)
        im_file = im_path[b]

        # fetch filter ids
        gst_data = Table.read(gst_file)
        filter_cols = [c for c in gst_data.colnames if "VEGA" in c]

        # extract every filter mentioned in the table
        filters = [f.split("_")[0] for f in filter_cols]

        # match with the gst filter list
        filter_ids = [gst_filter_names.index(i) for i in filters]

        filter_ids = [int(i) for i in filter_ids]

        gst_filter_names = [gst_filter_names[i] for i in filter_ids]
        beast_filter_names = [beast_filter_names[i] for i in filter_ids]

        print(beast_filter_names)

        # region file with catalog stars
        # make_region_file(gst_file, ref_filter[b])

        # -----------------
        # 0. make beast settings file
        # -----------------

        print("")
        print("creating beast settings file")
        print("")

        beast_settings_filename = create_beast_settings(
            gst_file,
            ast_input_file,
            gst_filter_names,
            beast_filter_names,
            galaxy,
            ref_image=im_file,
            supp=supp
        )

        # load in beast settings to get number of subgrids
        settings = beast_settings.beast_settings(
            beast_settings_filename
            #"beast_settings_" + galaxy + "_asts_" + field_names[b] + ".txt"
        )

        # -----------------
        # 1a. make magnitude histograms
        # -----------------

        print("")
        print("making magnitude histograms")
        print("")

        # if not os.path.isfile('./data/'+field_names[b]+'.gst_maghist.pdf'):
        peak_mags = plot_mag_hist.plot_mag_hist(gst_file, stars_per_bin=70, max_bins=75)

        # -----------------
        # 1b. make a source density map
        # -----------------

        print("")
        print("making source density map")
        print("")

        # not currently doing background density bins
        # use_bg_info = True
        use_bg_info = False
        if use_bg_info:
            background_args = types.SimpleNamespace(
                subcommand="background",
                catfile=gst_file,
                erode_boundary=settings.ast_erode_selection_region,
                pixsize=5,
                npix=None,
                reference=im_file,
                mask_radius=10,
                ann_width=20,
                cat_filter=[ref_filter, "90"],
            )

            create_background_density_map.main_make_map(background_args)

        # but we are doing source density bins!
        #if not os.path.isfile(gst_file.replace(".fits", "_source_den_image.fits")):
        print("No sd image file found")
        # - pixel size of 10 arcsec
        # - use ref_filter[b] between vega mags of 17 and peak_mags[ref_filter[b]]-0.5
        sourceden_args = types.SimpleNamespace(
            subcommand="sourceden",
            catfile=gst_file,
            erode_boundary=settings.ast_erode_selection_region,
            pixsize=5,
            npix=None,
            mag_name=ref_filter[0] + "_VEGA",
            mag_cut=[17, peak_mags[ref_filter[0]] - 0.5],
            flag_name=flag_filter[0] + "_FLAG",
        )
        create_background_density_map.main_make_map(sourceden_args)

        # new file name with the source density column
        gst_file_sd = gst_file.replace(".fits", "_with_sourceden.fits")

        # -----------------
        # 2. make physics model
        # -----------------

        print("")
        print("making physics model")
        print("")

        # see which subgrid files already exist
        gs_str = ""
        if settings.n_subgrid > 1:
            gs_str = "sub*"

        # It seems like there's a repeat call for the same physics file
        # but with a different variable name. We should probably get rid of
        # this duplicate since it's only used once.

        #sed_files = glob.glob(
        #    "./{0}/{0}_seds.grid{1}.hd5".format(field_names[b], gs_str)
        #)

        # And replace sed_files with model_grid_files

        # list of SED files (physics models)
        model_grid_files = sorted(
            glob.glob(
                "./{0}/{0}_seds.grid*.hd5".format(
                    field_names[b],
                )
            )
        )

        print(model_grid_files)

        # only make the physics model they don't already exist
        if len(model_grid_files) < settings.n_subgrid:
            # directly create physics model grids
            create_physicsmodel.create_physicsmodel(
                settings, nprocs=1, nsubs=settings.n_subgrid
            )

        # -------------------
        # 3. make AST inputs
        # -------------------

        print("")
        print("making AST inputs")
        print("")

        # only create an AST input list if the ASTs don't already exist
        if not os.path.isfile(ast_input_file):
            make_ast_inputs.make_ast_inputs(settings, pick_method="flux_bin_method")

        # compare magnitude histograms of ASTs with catalog
        plot_ast_histogram.plot_ast_histogram(
            ast_file=ast_input_file, sed_grid_file=model_grid_files[0]
        )

        if supp != 0:

            print("")
            print("making supplemental AST inputs")
            print("")

            ast_input_supp_file = "./{0}/{0}_inputAST_suppl.txt".format(field_names[b])

            if not os.path.isfile(ast_input_supp_file):
                make_ast_inputs.make_ast_inputs(settings, pick_method="suppl_seds")

        print("now go check the diagnostic plots!")


def input_ast_bin_stats(settings, ast_input_file, field_name):

    # Load input ast file
    ast_input = Table.read(ast_input_file, format="ascii")

    # Set reference and source density images
    reference_image = settings.ast_reference_image
    source_density_image = settings.obsfile.replace(".fits", "_source_den_image.fits")

    # Check stats
    map_file = settings.ast_density_table
    bdm = BinnedDensityMap.create(
        map_file,
        bin_mode=settings.sd_binmode,
        N_bins=settings.sd_Nbins,
        bin_width=settings.sd_binwidth,
        custom_bins=settings.sd_custom,
    )

    # Add RA and Dec information to the input AST file (which is just an ascii filewith only X,Y positions)
    hdu_ref = fits.open(reference_image)
    wcs_ref = WCS(hdu_ref[0].header)
    source_astin = wcs_ref.wcs_pix2world(ast_input["X"], ast_input["Y"], 0)

    # Compute source coordinates in SD image frame
    hdu_sd = fits.open(source_density_image)
    wcs_sd = WCS(hdu_sd[0].header)
    source_sdin = wcs_sd.wcs_world2pix(source_astin[0], source_astin[1], 0)

    # Import filter information from the BEAST settings file
    filters = settings.filters.copy()
    # Count number of filters and decide how many rows to plot
    ncmds = len(filters)
    nrows = int(ncmds / 2) + 1

    # Figure out what the bins are
    bin_foreach_source = np.zeros(len(ast_input), dtype=int)
    for i in range(len(ast_input)):
        bin_foreach_source[i] = bdm.bin_for_position(
            source_astin[0][i], source_astin[1][i]
        )
    # compute the AST input indices for each bin
    binnrs = np.unique(bin_foreach_source)
    bin_idxs = []
    for b in binnrs:
        sources_for_bin = bin_foreach_source == b
        bin_idxs.append([sources_for_bin])

    for k in range(len(binnrs)):
        cat = ast_input[bin_idxs[k]]
        print(binnrs[k], np.shape(cat["zeros"]))


def create_beast_settings(
    gst_file,
    ast_file,
    gst_filter_label,
    beast_filter_label,
    galaxy,
    ref_image="None",
    supp= 0
):
    """
    Create a beast_settings file for the given field.  This will open the file to
    determine the filters present - the `*_filter_label` inputs are references
    to properly interpret the file's information.

    Parameters
    ----------
    gst_file : string
        the path+name of the GST file

    ast_file : string
        the path+name of the AST file

    gst_filter_label : list of strings
        Labels used to represent each filter in the photometry catalog

    beast_filter_label : list of strings
        The corresponding full labels used by the BEAST

    galaxy: string
        The target galaxy (e.g., 'SMC', 'LMC')

    ref_image : string (default='None')
        path+name of image to use as reference for ASTs

    supp : integer (default=False)
        How many supplemental

    Returns
    -------
    nothing

    """

    # read in the catalog
    cat = Table.read(gst_file)
    # extract field name
    field_name = gst_file.split("/")[-1].split(".")[0]

    # get the list of filters
    filter_list_base = []
    filter_list_long = []
    for f in range(len(gst_filter_label)):
        filt_exist = [gst_filter_label[f] in c for c in cat.colnames]
        if np.sum(filt_exist) > 0:
            filter_list_base.append(gst_filter_label[f])
            filter_list_long.append(beast_filter_label[f])

    # read in the template settings file
    if galaxy == "SMC":
        template_file = "beast_settings_template_SMC_asts_custom.txt"
    if galaxy == "LMC":
        template_file = "beast_settings_template_LMC_asts_custom.txt"

    orig_file = open(template_file, "r")
    settings_lines = np.array(orig_file.readlines())
    orig_file.close()

    # write out an edited beast_settings
    beast_settings_filename = gst_file.replace("{0}.vgst.fits".format(field_name), "beast_settings_" + galaxy + "_asts_" + field_name + ".txt")
    new_file = open(beast_settings_filename, "w")

    for i in range(len(settings_lines)):

        # replace project name with the field ID
        if settings_lines[i][0:10] == "project = ":
            new_file.write('project = "' + field_name + '" \n')
        # obsfile
        elif settings_lines[i][0:10] == "obsfile = ":
            new_file.write('obsfile = "' + gst_file + '"\n')
        # AST file name
        elif settings_lines[i][0:10] == "astfile = ":
            new_file.write('astfile = "' + ast_file + '"\n')
        # BEAST filter names
        elif settings_lines[i][0:10] == "filters = ":
            new_file.write("filters = ['" + "','".join(filter_list_long) + "'] \n")
        # catalog filter names
        elif settings_lines[i][0:14] == "basefilters = ":
            new_file.write("basefilters = ['" + "','".join(filter_list_base) + "'] \n")
        # AST stuff
        elif settings_lines[i][0:20] == "ast_density_table = ":
            new_file.write(
                'ast_density_table = "'
                + gst_file.replace(".fits", "_sourceden_map.hd5")
                + '" \n'
            )
        elif settings_lines[i][0:22] == "ast_reference_image = ":
            new_file.write('ast_reference_image = "' + ref_image + '" \n')

        # supplemental AST stuff
        elif settings_lines[i][0:17] == "ast_supplement = ":
            if supp != 0:
                new_file.write('ast_supplement = True \nast_N_supplement = ' + str(supp) + '\n')

        #elif settings_lines[i][0:20] == "ast_existing_file = ":
        #    if supp != 0:
        #        new_file.write('ast_existing_file = "' + ast_file.replace(".txt", "_seds.txt") + '"\n')

        # none of those -> write line as-is
        else:
            new_file.write(settings_lines[i])

    new_file.close()

    return beast_settings_filename


def make_region_file(gst_file, ref_filter):
    """
    Make a region file out of the catalog file
    """

    with open(gst_file.replace(".fits", ".reg"), "w") as ds9_file:
        ds9_file.write(
            'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n'
        )
        ds9_file.write("fk5\n")

        cat = Table.read(gst_file)
        for i in range(len(cat)):
            # if cat['F555W_RATE'][i] != 0:
            # if cat[flag_filter[b]+'_FLAG'][i] < 99:
            if cat[ref_filter + "_VEGA"][i] < 26.65:
                ds9_file.write(
                    "circle("
                    + Angle(cat["RA"][i], u.deg).to_string(unit=u.hour, sep=":")
                    + ","
                    + Angle(cat["DEC"][i], u.deg).to_string(unit=u.deg, sep=":")
                    + ',0.1")\n'
                )
            else:
                ds9_file.write(
                    "circle("
                    + Angle(cat["RA"][i], u.deg).to_string(unit=u.hour, sep=":")
                    + ","
                    + Angle(cat["DEC"][i], u.deg).to_string(unit=u.deg, sep=":")
                    + ',0.1") # color=magenta \n'
                )


if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "field_name",
        type=str,
        help="name of target field",
    )
    parser.add_argument(
        "--ref_image",
        type=str,
        default=None,
        help="path to reference image",
    )
    parser.add_argument(
        "--filter_ids",
        type=list,
        default=None,
        help="indexes of filters",
    )
    parser.add_argument(
        "--galaxy",
        type=str,
        default=None,
        help="target galaxy",
    )

    parser.add_argument(
        "--supp",
        type=int,
        default=None,
        help="add N supplemental ASTs",
    )

    args = parser.parse_args()

    beast_ast_inputs(
        field_name=args.field_name,
        ref_image=args.ref_image,
        filter_ids=args.filter_ids,
        galaxy=args.galaxy,
        supp=args.supp
    )
