import os
import shutil

plots = os.listdir('method-paper-plots')

method_image_loc = '../papers/method_paper/images__method'
results_image_loc = '../papers/method_paper/images__results'
method_im = os.listdir(method_image_loc)
results_im = os.listdir(results_image_loc)

for im in method_im:
    if im in plots:
        print('{} -> {}'.format(
            os.path.join('method-paper-plots', im),
            os.path.join(method_image_loc, im),
        ))
        shutil.copyfile(
            os.path.join('method-paper-plots', im),
            os.path.join(method_image_loc, im),
        )

for im in results_im:
    if im in plots:
        print('{} -> {}'.format(
            os.path.join('method-paper-plots', im),
            os.path.join(results_image_loc, im),
        ))
        shutil.copyfile(
            os.path.join('method-paper-plots', im),
            os.path.join(results_image_loc, im),
        )
