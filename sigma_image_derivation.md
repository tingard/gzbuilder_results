# How to properly calculate sigma images for stacked frames in SDSS:

We want a way of taking multiple frames containing (at least part of) a target extended source, and obtaining an image and a sigma image in units of nanomaggies. Simply averaging sigma images of individual frames in quadrature does not properly account for possible covariances, proposed here is a potentially more thorough solution.

## In Theory

For each pixel, we have

$$\frac{I}{C} = \frac{n}{g} - S + V,$$

where $I$ represents the sky-subtracted, corrected image (nanomaggies), $C$ reprents the calibration image, $n$ is the number of electrons captured, $g$ is the gain, $S$ is the Sky value (data units) and $V$ is the dark current, $V = 0 ± \sqrt{v}$ ($v$ being the dark variance).

> Coleman: *SDSS lumps in the read-out noise (the thermal noise in the wires of the electronics on the telescope) with the dark variance... [this is a good reference textbook](http://hildaandtrojanasteroids.net/wrccd22oct06.pdf). Bias is a zero sec exposure, dark is a 0 light exposure of the same length as the observation. These are combined with the "read noise" to form the dark variance provided by SDSS*


Given Poisson error,

$$\sigma_n = \sqrt{n}.$$

If we stack images, given $N$ images of a pixel

$$n_\mathrm{total} = \sum_i{n_i} = \sum_i g_i\left(\frac{I_i}{C_i} + S_i - V_i\right),$$

$$ = \sum_{i}\frac{g_i}{C_i}I_i + \sum_i{g_i \left(S_i - V_i\right)} = \sigma_{n_\mathrm{total}}^2.$$

This is ideal, and is the level that many fitting software packages work at, we, however, want to return to working in units of nanomaggies on a stacked image, and so further calculation is needed:

$$I = \frac{1}{N}\sum_i I_i,$$

$$I = \frac{1}{N}\sum_i C_i\left(\frac{n_i}{g_i} - S_i + V_i\right),$$

And so 
$$\sigma_I^2 = \frac{1}{N^2}\sum_i\frac{C_i^2}{g_i^2}\sigma_{n_i}^2 + \frac{1}{N^2}\sum_i C_i^2 \sigma_{S_i}^2 + \frac{1}{N^2}\sum_i C_i^2 \sigma_{V_i}^2.$$

We treat the sky value as a constant, such that $\sigma_{S_i}^2 = 0$. Substituting $\sigma_{n_i}^2 = n_i$ as above gives

$$\sigma_I^2 = \frac{1}{N^2}\sum_i\frac{C_i^2}{g_i^2}n_i + \frac{1}{N^2}\sum_i C_i^2 v_i.$$

$$\sigma_I = \frac{1}{N}\sqrt{\sum_i C_i^2\left(\frac{n_i}{g_i^2} + v_i\right)}.$$

Note that this is identical to saying

$$\sigma_I^2 = \frac{1}{N^2}\sum_i\sigma_{I_i}^2.$$

## Weighted stacking

A (potentially) better way of stacking images would be to use a weighted average for pixel values, in which case

$$I = \frac{\sum_i \sigma_{I_i}^{-2}I_i}{\sum_i \sigma_{I_i}^{-2}}$$

The standard error of the weighted mean is thus

$$\sigma_{I} = \sqrt{\frac{1}{\sum_i \sigma_{I_i}^{-2}}}$$

$$\sigma_{I} = \left[\sum_i C_i^2\left(\frac{n_i}{g_i^2} + v_i\right)\right]^{-\frac{1}{2}}$$

*(this has not been implemented)*

## In Practise

- Our frames are not aligned.
- Calculate $n_i$, the electron counts for each frame.
- For each frame, create a slightly larger than required cutout of electron counts ($n_i$) and calibration images ($C_i$).
- Use `reproject` to align the electron counts and the calibration images of each frame to the WCS of the FITS header of the `Montage`-created image (which is what volunteer models were drawn on).
- Create the exact cutout we want of the reprojected electron counts, calibration image and sky image.
- Proceed with the above calculation.
	- Note that $N$ will not be the same for each pixel, as some regions of the image may be covered by different numbers of frames.
- We now have $\bar{I}$ and $\sigma_I$ 😁.
