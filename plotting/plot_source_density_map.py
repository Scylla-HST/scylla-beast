from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib as mpl

from beast.tools import beast_settings

def plot_source_density_map(sd_image_file, beast_settings_file):
    """
    Make a plot of the source density.
    
    The spatial bins are split into 5 arcsec^2. 
    Source density bins are determined by the specified beast settings file.
    The histogram bins are set by the bins originally used to create the ASTs
    (using the flux bin method), which are saved in
    ast_file.replace('inputAST','ASTfluxbins')
    and are automatically read in.

    Output plot is saved in the same location/name as image file, but with a .png
    instead of .fits.

    Parameters
    ----------
    sd_image_file : string
        name of SD image FITS file
 
    beast_settings_file : string
        name of beast settings .txt file


    """

    image_file = fits.open(sd_image_file)
    image_file.info()
    
    # assuming the image data is first 
    image_data = image_file[0].data
    
    image_file.close()
    
    # read in beast settings file
    settings = beast_settings.beast_settings(beast_settings_file)
    
    if settings.sd_binmode == "custom":
        sd_bins = [0] + settings.sd_custom
    
    # throw error if binning isn't custom    
    else: 
        raise Exception('Expected custom binning. Please ensure the right beast settings file is specified.')
    

    # define colormap
    cmap = plt.cm.viridis

    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    norm = mpl.colors.BoundaryNorm(sd_bins, cmap.N)

    fig = plt.figure(0, [10,10])
    im = plt.imshow(image_data, origin="lower", cmap=cmap, norm=norm)

    plt.colorbar(im,fraction=0.046, pad=0.04, ticks=sd_bins)

    plt.xlabel("Pixel (originally RA)")
    plt.ylabel("Pixel (originally DEC)")
    plt.title(r"Density of Sources per 5 arcsec$^2$")
    
    plt.tight_layout()

    fig.savefig(sd_image_file.replace("_image.fits", "_map_plot.png"))
    plt.close(fig)
    


if __name__ == "__main__":  # pragma: no cover

    parser = argparse.ArgumentParser()
    parser.add_argument("sd_image_file", type=str, help="name of Source Density image FITS file")
    parser.add_argument(
        "--beast_settings_file", type=str, help="name of beast settings file"
    )

    args = parser.parse_args()

    plot_source_density_map(args.sd_image_file, beast_settings_file=args.beast_settings_file)
