import fdcomp
import rasterio
import numpy as np

wd = '/Volumes/Cloud/Yandex/RSCF_Flow_directions/TESTS_raster_generalization'

acc1_src = rasterio.open(f'{wd}/HydroSHEDS/hyd_glo_acc_30s_sa.tif')
acc1 = acc1_src.read(1)

dir1_src = rasterio.open(f'{wd}/HydroSHEDS/hyd_glo_dir_30s_sa.tif')
dir1 = dir1_src.read(1)
aff1 = np.array(dir1_src.get_transform())[[1, 2, 0, 4, 5, 3]]

dir2_src = rasterio.open(f'{wd}/COTAT/COTAT_05_flowdir_sa.tif')
dir2 = dir2_src.read(1)
aff2 = np.array(dir2_src.get_transform())[[1, 2, 0, 4, 5, 3]]

tree, seeds = fdcomp.d8tree(acc1, dir1, aff1)
accuracy = fdcomp.d8comp(dir1, dir2, seeds[219:220])
