import numpy as np
import glob
import subprocess
import sys
import os
import shutil
import types
import re

from astropy.wcs import WCS

from beast.tools.run import (
    create_physicsmodel,
    make_ast_inputs,
    create_obsmodel,
    make_trim_scripts,
    run_fitting,
    merge_files,
    create_filenames,
)

from beast.tools import (
    beast_settings,
    create_background_density_map,
    subdivide_obscat_by_source_density,
    split_catalog_using_map,
    cut_catalogs,
    write_sbatch_file,
    setup_batch_beast_fit,
    # star_type_probability,
    # compare_spec_type,
    reorder_beast_results_spatial,
    condense_beast_results_spatial,
)
from beast.plotting import (
    plot_mag_hist,
    make_ds9_region_file,
    plot_chi2_hist,
    plot_cmd_with_fits,
    plot_triangle,
    plot_indiv_pdfs,
    # plot_completeness,
)


from astropy.table import Table
from astropy.coordinates import Angle
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits


def beast_production_wrapper():
    """
    This does all of the steps for a full production run, and can be used as
    a wrapper to automatically do most steps for multiple fields.
    1. make source density and background maps
    2. make beast settings file
    3. make sbatch script for physics model (SED grid)
    4. make quality cuts to photometry & fake stars
    5. split catalog by either source density or background
    6. make sbatch script for noise model
    7. make sbatch script for trimming models
    8. make sbatch script for fitting models
    9. make sbatch script to merge output files
    10. make sbatch script for running some analysis
    """

    # set xsede username
    username = "clindber"

    # get the list of Scylla fields
    scylla_info = Table.read("scylla_fields.txt", format="ascii")

    field_names = scylla_info["field"]
    ref_filter = ['F475W']
    flag_filter = ['F475W']

    n_field = len(field_names)

    # catalog and the BEAST filter names.  Each filter has a column in
    # scylla_info to tell what the corresponding BEAST filter name is for each
    # image.
    gst_filter_names = [
        "F225W",
        "F275W",
        "F336W",
        "F475W",
        "F657N",
        "F814W",
        "F110W",
        "F160W",
    ]

    beast_filter_names = [
        "HST_WFC3_F225W",
        "HST_WFC3_F275W",
        "HST_WFC3_F336W",
        "HST_WFC3_F475W",
        "HST_WFC3_F657N",
        "HST_WFC3_F814W",
        "HST_WFC3_F110W",
        "HST_WFC3_F160W",
    ]

    # keep track of what needs to be submittted
    sbatch_list = []

    for b in range(n_field):

        print("********")
        print("field " + field_names[b])
        print("********\n")

        # -----------------
        # data file names
        # -----------------

        # paths for the data/AST files
        gst_file = "./data/" + field_names[b] + ".vgst.fits"
        ast_file = "./data/" + field_names[b] + ".vgst.fake.fits"
        
        
        if not os.path.isfile(ast_file):
            print("no AST file for this field")
            continue

        ## - ref image
        #im_file = "./data/" + field_names[b] + "_" + ref_filter[b] + "_drz.chip1.fits"

        # -----------------
        # 1a. make magnitude histograms
        # -----------------

        #peak_mags = plot_mag_hist.plot_mag_hist(gst_file, stars_per_bin=70, max_bins=75)

        # -----------------
        # 1b. retrieve source density map
        # -----------------
        # load catalog
        gst_data = Table.read(gst_file)

        if not os.path.exists(gst_file.replace(".fits","_with_sourceden.fits")):
            # source density map
            sd_map = gst_file.replace(".fits", "_source_den_image.fits")

            # write new catalog with source density column
            sd = fits.open(sd_map)
            wcs = WCS(sd[0].header)
            sd_pix = create_background_density_map.get_pix_coords(gst_data, wcs)
            source_densities = sd[0].data[np.array(sd_pix[1], dtype='int'), np.array(sd_pix[0], dtype='int')]
            gst_data['SourceDensity'] = source_densities
            gst_data.write(gst_file.replace(".fits", "_with_sourceden.fits"))

        # new file name with the source density column
        gst_file_sd = gst_file.replace(".fits", "_with_sourceden.fits")

        # -----------------
        # 2. make beast settings file
        # -----------------

        print("")
        print("creating beast settings file")
        print("")
 
        # fetch filter ids from obs table
        filter_cols = [c for c in gst_data.colnames if "VEGA" in c]

        # extract every filter mentioned in the table
        filters = [f.split('_')[0] for f in filter_cols]

        # match with the gst filter list
        filter_ids = [gst_filter_names.index(i) for i in filters]
        filter_ids.sort()

        filter_sublist = [gst_filter_names[i] for i in filter_ids]
        beast_filter_sublist = [beast_filter_names[i] for i in filter_ids]

        settings_file = create_beast_settings(
            gst_file,
            ast_file,
            filter_sublist,
            beast_filter_sublist)

        # load in beast settings to get number of subgrids
        settings = beast_settings.beast_settings(settings_file)

        # make a project directory
        proj_dir = "./{0}/".format(settings.project)
        if not os.path.isdir(proj_dir):
            os.mkdir(proj_dir)

        # -----------------
        # 3. make physics model
        # -----------------

        print("")
        print("making physics model")
        print("")

        # master SED files
        if "LMC" in gst_file:
            master_sed_files = [
                "/ocean/projects/ast190023p/shared/scylla/mastergrid_LMC/mastergrid_LMC_seds.gridsub" + str(i) + ".hd5"
                for i in range(settings.n_subgrid)
            ]
        if "SMC" in gst_file:
            master_sed_files = [
                "/ocean/projects/ast190023p/shared/scylla/mastergrid_SMC/mastergrid_SMC_seds.gridsub" + str(i) + ".hd5"
                for i in range(settings.n_subgrid)
            ]

        if not os.path.isfile(master_sed_files[0]):
            make_mastergrid()

        # need to have list of the model grid file names to make
        model_grid_files = [
            "./{0}/{0}_seds.gridsub{1}.hd5".format(settings.project, i)
            for i in range(settings.n_subgrid)
        ]

        # copy from the master
        for i, sed_file in enumerate(model_grid_files):

            # grid doesn't exist -> script to copy over from master grid
            if not os.path.isfile(sed_file):

                # file name for script
                fname = "./{0}/model_batch_jobs/copy_gridsub{1}.script".format(
                    settings.project, i
                )

                # make sure the directory exists
                fname_path = os.path.dirname(fname)
                if not os.path.isdir(fname_path):
                    os.mkdir(fname_path)
                log_path = os.path.join(fname_path, "logs")
                if not os.path.isdir(log_path):
                    os.mkdir(log_path)

                # write out the sbatch file
                write_sbatch_file.write_sbatch_file(
                    fname,
                    (
                        "python -m beast.tools.remove_filters {0}"
                        " --physgrid {1} --physgrid_outfile {2} "
                        " --beast_filt {3}"
                        " >> {4}/rem_filt_{5}.log".format(
                            gst_file,
                            master_sed_files[i],
                            sed_file,
                            " ".join(beast_filter_sublist),
                            log_path,
                            i,
                        )
                    ),
                    "/ocean/projects/ast190023p/" + username + "/scylla",
                    modules=["module load anaconda3", "source activate beast_prod"],
                    stdout_file="/ocean/projects/ast190023p/{0}/scylla/{1}/model_batch_jobs/logs/%j.out".format(
                        username, settings.project
                    ),
                    queue="EM",
                    nodes="24",
                    run_time="10:00:00",
                    #mem="{0:.0f}GB".format(
                    #    os.path.getsize(master_sed_files[i]) / 10 ** 9 * 2.9
                    #),
                )
                sbatch_list.append("sbatch " + fname)
                print("sbatch " + fname)

        # make a file that contains the SED file names
        with open("./{0}/subgrid_fnames.txt".format(settings.project), "w") as fname:
            for sed_file in model_grid_files:
                fname.write(sed_file + "\n")

        # now let's check if we needed to make any more
        sed_files = glob.glob(
            "./"
            + field_names[b]
            + "_beast/"
            + field_names[b]
            + "_beast_seds.gridsub*.hd5"
        )
        if len(sed_files) < settings.n_subgrid:
            print("\n**** go run physics model code for " + field_names[b] + "! ****")
            continue

        # -----------------
        # make ASTs
        # -----------------

        # -- ALREADY DONE --

        # -----------------
        # 4. edit photometry/AST catalogs
        # -----------------

        # remove sources that are
        # - in regions without full imaging coverage,
        # - flagged in flag_filter

        print("")
        print("editing photometry/AST catalogs")
        print("")

        gst_to_use = gst_file_sd
        gst_file_cut = gst_file.replace(".fits", "_with_sourceden_cut.fits")

        ast_file_cut = ast_file.replace(".fits", "_cut.fits")

        cut_catalogs.cut_catalogs(
            gst_to_use,
            gst_file_cut,
            ast_file,
            ast_file_cut,
            partial_overlap=True,
            flagged=True,
            flag_filter=flag_filter[b],
            region_file=True,
        )

        # edit the settings file to have the correct photometry file name
        settings_file = create_beast_settings(
            gst_file_cut,
            ast_file_cut,
            filter_sublist,
            beast_filter_sublist,
        )

        # load in beast settings to get number of subgrids
        settings = beast_settings.beast_settings(settings_file)

        # -----------------
        # 5. split observations
        # -----------------

        bin_table = 'sourceden'

        print("")
        print("splitting observations by " + bin_table)
        print("")
    
        min_n_subfiles=5

        # check if any trimmed gst_files have less than 5 sources
        #for i, gfile in enumerate(gst_file_cut):
        #    file_len = len(Table.read(gfile))
        #    if file_len < min_n_subfiles:
        #        min_n_subfiles=file_len
        #        print("setting min_n_subfiles = "+file_len)

        split_catalog_using_map.split_main(
            settings,
            gst_file_cut,
            ast_file_cut,
            gst_file.replace(".fits", "_" + bin_table + "_map.hd5"),
            n_per_file=1000,
            min_n_subfile=min_n_subfiles,
        )

        # check for pathological cases of AST bins not matching photometry bins
        gst_list = glob.glob(gst_file_cut.replace(".fits", "*bin?.fits"))
        ast_list = glob.glob(ast_file_cut.replace(".fits", "*bin?.fits"))
        if len(gst_list) != len(ast_list):
            for a in ast_list:
                # the bin number for this AST file
                bin_num = a[a.rfind("_") + 1 : -5]
                # if this bin number doesn't have a corresponding gst file,
                # delete the AST bin file
                if np.sum([bin_num in g for g in gst_list]) == 0:
                    print("removing " + a)
                    os.remove(a)

        # -- at this point, we can run create_filenames to make final lists of filenames
        file_dict = create_filenames.create_filenames(
            settings, use_sd=True, nsubs=settings.n_subgrid
        )

        # figure out how many files there are
        sd_sub_info = file_dict["sd_sub_info"]
        # - number of SD bins
        temp = set([i[0] for i in sd_sub_info])
        print("** total SD bins: " + str(len(temp)))
        # - the unique sets of SD+sub
        unique_sd_sub = [
            x for i, x in enumerate(sd_sub_info) if i == sd_sub_info.index(x)
        ]
        print("** total SD subfiles: " + str(len(unique_sd_sub)))

        # -----------------
        # 6. make noise models
        # -----------------

        print("")
        print("making noise models")
        print("")

        # expected final list of noise files
        noisefile_list = list(set(file_dict["noise_files"]))
        # the current existing ones (if any)
        existing_noisefiles = glob.glob(
            "{0}/{0}_noisemodel_bin*.gridsub*.hd5".format(settings.project)
        )

        # if we don't have all of them yet, write script to make them
        if len(existing_noisefiles) < len(noisefile_list):

            noise_dir = "./{0}/noise_logs".format(settings.project)
            if not os.path.isdir(noise_dir):
                os.mkdir(noise_dir)

            cmd = (
                f"python -m beast.tools.run.create_obsmodel {settings_file} "
                + f"--nsubs {settings.n_subgrid} --use_sd --nprocs 1 "
                + "--subset ${SLURM_ARRAY_TASK_ID} $((${SLURM_ARRAY_TASK_ID} + 1)) "
                + f">> {noise_dir}/create_noisemodel_${{SLURM_ARRAY_TASK_ID}}.log"
            )
            fname = f"{settings.project}/create_noisemodels.script"

            write_sbatch_file.write_sbatch_file(
                fname,
                cmd,
                "/ocean/projects/ast190023p/" + username + "/scylla",
                modules=["module load anaconda3", "source activate beast_prod"],
                job_name="beast_LH",
                stdout_file="{0}/%A_%a.out".format(os.path.abspath(noise_dir)),
                queue="EM",
                run_time="20:00:00",
                #mem="{0:.0f}GB".format(
                #    os.path.getsize(file_dict["modelsedgrid_files"][0]) / 10 ** 9 * 3
                #),
                array=[0, settings.n_subgrid - 1],
            )

            sbatch_list.append("sbatch " + fname)

            print(
                "*** go run {0}/create_noisemodels.script ***".format(settings.project)
            )

            continue

        # plot completeness
        if False:
            print("plotting completeness")
            plot_completeness.plot_completeness(
                file_dict["modelsedgrid_files"][0],
                file_dict["noise_files"][0],
                field_names[b] + "_completeness.pdf",
            )

        # -----------------
        # 7. make script to trim models
        # -----------------

        print("")
        print("setting up script to trim models")
        print("")

        job_file_list = make_trim_scripts.make_trim_scripts(
            settings, num_subtrim=1, prefix=None
        )

        if len(job_file_list) > 0:
            print("\n**** go run trimming code for " + field_names[b] + "! ****")

            fname = "{0}/trim_files.script".format(settings.project)

            write_sbatch_file.write_sbatch_file(
                fname,
                '{0}/trim_batch_jobs/BEAST_gridsub"${{SLURM_ARRAY_TASK_ID}}"_batch_trim.joblist'.format(
                    settings.project
                ),
                "/ocean/projects/ast190023p/" + username + "/scylla",
                modules=["module load anaconda3", "source activate beast_prod"],
                job_name="beast_LH",
                stdout_file="/ocean/projects/ast190023p/{0}/scylla/{1}/trim_batch_jobs/logs/%A_%a.out".format(
                    username, settings.project
                ),
                queue="EM",
                run_time="20:00:00",
                array=[0, len(job_file_list) - 1],
            )

            print(f"sbatch {fname}")
            sbatch_list.append("sbatch " + fname)

            continue

        else:
            print("all files are trimmed for " + field_names[b])

        # -----------------
        # 8. make script to fit models
        # -----------------

        print("")
        print("setting up script to fit models")
        print("")

        if True:

            fit_run_info = setup_batch_beast_fit.setup_batch_beast_fit(
                settings,
                num_percore=1,
                overwrite_logfile=False,
                # pdf2d_param_list=['Av', 'Rv', 'f_A', 'M_ini', 'logA', 'Z', 'distance','logT', 'logg'],
                pdf2d_param_list=None, #["Av", "M_ini", "logT"],
                # prefix='source activate bdev',
                use_sd=True,
                nsubs=settings.n_subgrid,
                nprocs=1,
            )
            
            msz_base = 0.
            k=0
            for phot_file in fit_run_info["phot_file"]:
                #msz = os.path.getsize(phot_file)/ 10**9
                binno = phot_file.split('cut_bin')[1].split('_sub')[0]
                subno = phot_file.split('_sub')[1].split('.fits')[0]
                
                grid_file = './'+field_names[b]+'_beast/bin'+binno+'_sub'+subno+'/'+field_names[b]+'_beast_bin'+binno+'_sub'+subno+'_gridsub'+str(k)+'_seds_trim.grid.hd5'

                ngrid_file = './'+field_names[b]+'_beast/bin'+binno+'_sub'+subno+'/'+field_names[b]+'_beast_bin'+binno+'_sub'+subno+'_gridsub'+str(k)+'_noisemodel_trim.grid.hd5'
                msz = (os.path.getsize(grid_file)+os.path.getsize(ngrid_file)) / 10**9
                if k<9:
                    k+=1
                else:
                    k=0
                if msz > msz_base:
                    msz_base = msz
            print('max size: ', msz_base)

            # check if the fits exist before moving on
            tot_remaining = len(fit_run_info["done"]) - np.sum(fit_run_info["done"])
            if tot_remaining > 0:
                print("\n**** go run fitting code for " + field_names[b] + "! ****")

                fname = "{0}/run_fitting.script".format(settings.project)

                write_sbatch_file.write_sbatch_file(
                    fname,
                    '{0}/fit_batch_jobs/beast_batch_fit_"${{SLURM_ARRAY_TASK_ID}}".joblist'.format(
                        settings.project
                    ),
                    "/ocean/projects/ast190023p/" + username + "/scylla",
                    modules=["module load anaconda3", "source activate beast_prod"],
                    job_name="beast_LH",
                    stdout_file="/ocean/projects/ast190023p/{0}/scylla/{1}/fit_batch_jobs/logs/%A_%a.out".format(
                        username, settings.project
                    ),
                    queue="RM-shared",
                    run_time="20:00:00",
                    nodes="24",
                    array=[1, tot_remaining],
                )

                sbatch_list.append("sbatch " + fname)

                # also write out a file to do partial merging in case that
                # ends up being useful
                write_sbatch_file.write_sbatch_file(
                    "{0}/merge_files_partial.script".format(settings.project),
                    f"python -m beast.tools.run.merge_files {settings_file} --use_sd 1 --nsubs {settings.n_subgrid} --partial 1",
                    "/ocean/projects/ast190023p/" + username + "/scylla",
                    modules=["module load anaconda3", "source activate beast_prod"],
                    stdout_file="/ocean/projects/ast190023p/{0}/scylla/{1}/fit_batch_jobs/logs/%j.out".format(
                        username, settings.project
                    ),
                    queue="RM-shared",
                    run_time="2:00:00",
                    nodes="4",
                )

                continue
            else:
                print("all fits are complete for " + field_names[b])

        # -----------------
        # 9. merge stats files from each fit
        # -----------------

        print("")
        print("merging files")
        print("")

        # use the merged stats file to decide if merging is complete
        merged_stats_file = "{0}_beast/{0}_beast_stats.fits".format(field_names[b])

        if not os.path.isfile(merged_stats_file):

            # write out the sbatch file
            fname = "{0}/merge_files.script".format(settings.project)
            write_sbatch_file.write_sbatch_file(
                fname,
                f"python -m beast.tools.run.merge_files {settings_file} --use_sd 1 --nsubs {settings.n_subgrid}",
                "/ocean/projects/ast190023p/" + username + "/scylla",
                modules=["module load anaconda3", "source activate beast_prod"],
                stdout_file="/ocean/projects/ast190023p/{0}/scylla/{1}/fit_batch_jobs/logs/%j.out".format(
                    username, settings.project
                ),
                queue="RM-shared",
                run_time="2:00:00",
                nodes="4",
            )

            sbatch_list.append("sbatch " + fname)

            continue

        # -----------------
        # make some plots
        # -----------------

        # print('')
        # print('making some plots')
        # print('')

        # chi2 histogram
        # plot_chi2_hist.plot(stats_filebase+'_stats.fits', n_bins=100)
        # CMD color-coded by chi2
        # plot_cmd_with_fits.plot(gst_file, stats_filebase+'_stats.fits',
        #                            mag1_filter='F475W', mag2_filter='F814W', mag3_filter='F475W',
        #                            param='chi2min', log_param=True)

        #'F275W','F336W','F390M','F555W','F814W','F110W','F160W'

        # -----------------
        # reorganize results into spatial regions
        # -----------------

        # print('')
        # print('doing spatial reorganizing')
        # print('')

        region_filebase = './' + field_names[b] + '_beast/' + field_names[b] + '_beast_sd'
        output_filebase = './' + field_names[b] + '_beast/spatial/' + field_names[b]
        stats_filebase = './' + field_names[b] + '_beast/' + field_names[b] + '_beast'

        reorder_beast_results_spatial.reorder_beast_results_spatial(stats_filename=stats_filebase + '_stats.fits',
                                                                        region_filebase=region_filebase,
                                                                        output_filebase=output_filebase)

        condense_beast_results_spatial.condense_files(filedir='./' + field_names[b] + '_beast/spatial/')

        # -----------------
        # 10. some sciency things
        # -----------------

        print("")
        print("doing some science")
        print("")

        cmd_list = []

        # naive maps
        if not os.path.isfile(merged_stats_file.replace("stats", "mapAv")):
            cmd_list.append(
                f"python -m megabeast.make_naive_maps {merged_stats_file} --pix_size 10"
            )
        # naive IMF
        if not os.path.isfile("{0}/{0}_imf.pdf".format(settings.project)):
            cmd_list.append(
                f"python -m megabeast.make_naive_imf {settings_file} --use_sd 1 --compl_filter {ref_filter[b]}"
            )

        # if there are things to make, write out sbatch file
        if len(cmd_list) > 0:

            fname = "{0}/run_science.script".format(settings.project)

            write_sbatch_file.write_sbatch_file(
                fname,
                cmd_list,
                "/ocean/projects/ast190023p/" + username + "/scylla",
                modules=["module load anaconda3", "source activate beast_prod"],
                stdout_file="/pylon5/as5pi7p/{0}/scylla/{1}/fit_batch_jobs/logs/%j.out".format(
                    username, settings.project
                ),
                run_time="12:00:00",
                mem="128GB",
            )

            sbatch_list.append("sbatch " + fname)

    # write out all the sbatch commands to run
    with open("sbatch_commands.script", "w") as f:
        for cmd in sbatch_list:
            f.write(cmd + "\n")


def create_beast_settings(
    gst_file,
    ast_file,
    gst_filter_label,
    beast_filter_label,
):
    """
    Create a beast settings file for the given field.
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
    Returns
    -------
    settings_file : str
        name of settings file
    """

    # extract field name
    field_name = gst_file.split("/")[-1].split(".")[0]

    # get the list of filters
    filter_list_base = gst_filter_label
    filter_list_long = beast_filter_label

    # read in the template settings file
    if "SMC" in gst_file:
        orig_filename = "beast_settings_template_SMC.txt"
    if "LMC" in gst_file:
        orig_filename = "beast_settings_template_LMC.txt"
    with open(orig_filename, "r") as orig_file:
        settings_lines = np.array(orig_file.readlines())

    # write out an edited settings file
    settings_file = "beast_settings_" + field_name + ".txt"
    new_file = open(settings_file, "w")

    for i in range(len(settings_lines)):

        # replace project name with the field ID
        if settings_lines[i][0:10] == "project = ":
            new_file.write('project = "' + field_name + '_beast"\n')
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
        # none of those -> write line as-is
        else:
            new_file.write(settings_lines[i])

    new_file.close()

    return settings_file


def make_mastergrid():
    """
    Setup sbatch files to make master grids for LMC and SMC
    """

    for gal_name in ["LMC"]:

        settings_file = f"beast_settings_{gal_name}_mastergrid.txt"

        # read in the settings
        settings = beast_settings.beast_settings(settings_file)

        # make physics model scripts
        create_physicsmodel.split_create_physicsmodel(
            settings, nprocs=1, nsubs=settings.n_subgrid
        )

        # make an sbatch file for them
        write_sbatch_file.write_sbatch_file(
            f"create_{gal_name}_mastergrid.script",
            f'./mastergrid_{gal_name}/model_batch_jobs/create_physicsmodel_"${{SLURM_ARRAY_TASK_ID}}".job',
            "/ocean/projects/ast190023p/cmurray3/scylla",
            modules=["module load anaconda3", "source activate beast_prod"],
            job_name=f"{gal_name}grid",
            stdout_file=f"/ocean/projects/ast190023p/cmurray3/scylla/mastergrid_{gal_name}/model_batch_jobs/logs/%A_%a.out",
            egress=False,
            queue="EM",
            run_time="40:00:00",
            mem="570GB",
            array=[0, settings.n_subgrid - 1],
        )


if __name__ == "__main__":

    beast_production_wrapper()
