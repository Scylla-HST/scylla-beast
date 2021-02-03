Scylla HST
==========
Repository for code related to the HST program Scylla!

AST Workflow
--------------

In the ast_workflow/ folder, you can find the BEAST wrapper script and template files needed to generate artificial star test (AST) inputs for Scylla fields. It is based on the production_run_2019 workflows in the `BEAST-Fitting/beast-examples/ <https://github.com/BEAST-Fitting/beast-examples>`_ repo, with some additions specific to Scylla. To execute the script, you need to specify the field name (e.g., 15891_SMC-3956ne-9632) the reference image (e.g., data/15891_SMC-3956ne-9632_F475W_drc_wcs_ref_test.fits), the filters included in the field (done via index relative to the master list of Scylla filters: F225W, F275W, F336W, F475W, F814W, F110W, F160W, F657N; so a field with F336W, F475W and F814W would be 234) and finally the galaxy the target is in (determines which template AST file is used, so far only SMC has a template). So, for example, for the test field SMC-6:

.. code-block:: console

    $ python -m beast_ast_inputs 15891_SMC-3956ne-9632 \
          --ref_image data/15891_SMC-3956ne-9632_F475W_drc_wcs_ref_test.fits \
          --filter_ids 234 --galaxy ‘SMC'

This all assumes that you have the catalog file (e.g., '*.gst.fits') and the reference image ('*_ref.fits') for each field in the data/ folder, as well as the template settings file for each galaxy. Then the script runs through the nominal workflow up to the creation of ASTs. In detail, it uses the provided arguments to produce a custom beast_settings file for each field (where the params that change are the field name, reference file information and filter information — for each galaxy, the (coarse) physics model params and source density binning params are the same, and are drawn from the template settings file), makes magnitude histograms and a source density image, creates a physics grid, generates AST inputs replicated across source density bins, trims the AST inputs (mildly) to remove extremely faint sources. 
