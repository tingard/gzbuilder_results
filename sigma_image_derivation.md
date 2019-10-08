# How to properly calculate sigma images for stacked frames in SDSS:

We want a way of taking multiple frames containing (at least part of) a target extended source, and obtaining an image and a sigma image in units of nanomaggies. Simply averaging sigma images of individual frames in quadrature does not properly account for possible covariances, proposed here is a potentially more thorough solution.

## In Theory

For each pixel, we have

$$\frac{I}{C} = \frac{n}{g} - S,$$

where $I$ represents the sky-subtracted, corrected image (nanomaggies), $C$ reprents the calibration image, $n$ is the number of electrons captured, $g$ is the gain and $S$ is the Sky value (data units).

Given Poisson error,

$$\sigma_n = \sqrt{n}.$$

If we stack images, given $N$ images of a pixel

$$n_\mathrm{total} = \sum_i{n_i} = \sum_i g_i\left(\frac{I_i}{C_i} + S_i\right),$$

$$\sigma_{n_\mathrm{total}}^2 = \sum_{i}\frac{g_i}{C_i}I_i + \sum_i{g_i S_i}.$$


Meaning

$$\sum_i\frac{g_i}{C_i}I_i = \sum_i{n_i} - \sum_i{g_i S_i}.$$

Assuming $g_i$ and $C_i$ are nearly constant gives

$$\frac{\left<g\right>}{\left<C\right>}\sum_i{I_i} = \sum_i{n_i} -\left<g\right>\sum_i{S_i},$$ 

meaning that the average image in nanomaggies is given by

$$\bar{I} = \frac{1}{N} \sum_i I_i = \frac{\left<C\right>}{N\left<g\right>}\sum_i{n_i} - \frac{\left<C\right>}{N}\sum_i{S_i},$$

$$\bar{I} = \frac{1}{N} \sum_i I_i = \frac{\left<C\right>}{N}\left(\frac{n_\mathrm{total}}{\left<g\right>} - \sum_i{S_i}\right).$$

The corresponding error is therefore given by

$$\sigma_{I}^2 = \sum_i\left(\frac{\mathrm{d}\bar{I}}{\mathrm{d}n_i}\right)^2\sigma_{n_i}^2 + \sum_i\left(\frac{\mathrm{d}\bar{I}}{\mathrm{d}S_i}\right)^2\sigma_{S_i}^2,$$

$$\sigma_{I}^2 = \sum_i\left(\frac{1}{N}\frac{\left<C\right>}{\left<g\right>}\right)^2\sigma_{n_i}^2 + \sum_i\frac{\left<C\right>^2}{N^2}\sigma_{S_i}^2.$$

Substituting $\sigma_{n_i}^2 = n_i$ as above, and $\sigma_{S_i}^2 = v_i$ for the dark-variance of an image

$$\sigma_{I}^2 = \frac{\left<C\right>^2}{N^2}\left(\frac{1}{\left<g\right>^2}\sum_i{n_i} - \sum_i{v_i}\right),$$

$$\sigma_{I} = \sqrt{\frac{n_\mathrm{total}}{\left<g\right>^2} + \sum_i{v_i}}\;\frac{\left<C\right>}{N}.$$



## In Practise

- Our frames are not aligned.
- Calculate $n_i$, the electron counts for each frame
- For each frame, create a slightly larger than required cutout of nelec and the Calibration image
- Use `reproject` to align the electron counts and the calibration images of each frame to the WCS of the FITS header of the `Montage`-created image (which is what volunteer models were drawn on).
- Proceed with the above calculation
	- Note that $N$ will not be the same for each pixel, as some regions of the image may be covered by different numbers of frames
- Once we have $\bar{I}$ and $\sigma_I$, perform a cutout of the required size for each and return!

_**n.b.**_: *MAKE LOTS OF SAVE POINTS* (i.e. write out the cutout + reprojected FITS files)
