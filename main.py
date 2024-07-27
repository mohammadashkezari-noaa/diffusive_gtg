import shutil
from copy import deepcopy
from typing import Optional, Union
from osgeo import gdal
import numpy as np
import skimage as ski
gdal.UseExceptions()


class DiffusiveGTG():
    """
    Assumptions:
    - all subgrids have identical horizontal crs and is not projected
    - the NoDataValue is identical in all subgrids
    - the subgrids have only one band
    - subgrids are stacked from low to high resolution
    """
    def __init__(self, input_file: str) -> None:
        """
        Parameters
        ----------
        input_file: str
            Absolute path to the GTG file.
        """
        self.input_file = input_file
        self.gtg_metadata = self.get_metadata()
        self.n_subgrids = len(self.gtg_metadata["subgrids"]) // 2
        self.subgrids_metadata = [self.get_metadata(i) for i in range(1, self.n_subgrids+1)]
        self.compile_subgrid_arrays()        

    def get_metadata(self, subgrid_index: int = -1) -> Union[dict, str]:
        """
        Return a dictionary containing generic raster metadata
        such as number of bands, extents, etc.

        Parameters
        -----------
        subgrid_index: int
            subdataset index. If -1 the parent file's metadata is returned.
        """
        try:
            metadata = {}
            prefix = "" if subgrid_index == -1 else f"GTIFF_DIR:{subgrid_index}:"
            ds = gdal.Open(prefix + self.input_file, gdal.GA_ReadOnly)
            metadata |= {"description": ds.GetDescription()}
            metadata |= {"bands": ds.RasterCount}
            metadata |= {"dimensions": f"{ds.RasterXSize} x {ds.RasterYSize}"}
            metadata |= {"band_no_data": [ds.GetRasterBand(i+1).GetNoDataValue()
                                          for i in range(ds.RasterCount)]}
            metadata |= {"band_descriptions": [ds.GetRasterBand(i+1).GetDescription()
                                               for i in range(ds.RasterCount)]}
            geot = ds.GetGeoTransform()
            res_x, res_y = geot[1], geot[5]
            x_min, y_max = geot[0], geot[3]
            x_max = x_min + res_x * ds.RasterXSize
            y_min = y_max + res_y * ds.RasterYSize
            metadata |= {"geo_transform": geot}
            metadata |= {"extent": [x_min, y_min, x_max, y_max]}
            metadata |= {"resolution": [res_x, res_y]}
            ds = None
            if subgrid_index == -1:
                metadata |= {"subgrids": gdal.Info(self.input_file, format="json")["metadata"]["SUBDATASETS"]}
        except Exception as e:
            return f"Unable to get raster metadata: {e}"
        return metadata

    def compile_subgrid_arrays(self):
        self.subgrid_arrays = []
        for i in range(1, self.n_subgrids+1):
            ds = gdal.Open(f"GTIFF_DIR:{i}:{self.input_file}")
            self.subgrid_arrays.append(ds.GetRasterBand(1).ReadAsArray())
            ds = None
        return self.subgrid_arrays

    def _index_range(self, g_ind: int, sub_extent: list) -> Optional[list]:
        """
        Return the index range corresponding to a geographic extent, if valid.
        Otherwise return None.
        """
        if not sub_extent:
            return None
        e = self.subgrids_metadata[g_ind]["extent"]
        res_x, res_y = self.subgrids_metadata[g_ind]["resolution"]
        res_y = np.fabs(res_y)
        h = self.subgrid_arrays[g_ind].shape[0]
        index_range = [int((sub_extent[0] - e[0]) / res_x),
                       h - int((sub_extent[3] - e[1]) / res_y),
                       int((sub_extent[2] - e[0]) / res_x),
                       h - int((sub_extent[1] - e[1]) / res_y)
                       ]
        for ind in index_range:
            if ind < 0:
                return None
        return index_range

    def intersection(self, g_ind1: int, g_ind2: int) -> Optional[tuple]:
        """
        Return the coordinates of the overlapping area between two subgrids.
        Return None when there is no overlap.

        Parameters
        -----------
        g_ind1: int
            index of the first subgrid
        g_ind2: int
            index of the second subgrid
        """
        e1 = self.subgrids_metadata[g_ind1]["extent"]
        e2 = self.subgrids_metadata[g_ind2]["extent"]
        lon_max, lon_min = min(e1[2], e2[2]), max(e1[0], e2[0])
        lat_max, lat_min = min(e1[3], e2[3]), max(e1[1], e2[1])
        overlap = [lon_min, lat_min, lon_max, lat_max]
        if lon_min >= lon_max or lat_min >= lat_max:
            overlap = None
        return overlap

    def leak(self, src: int):
        """
        Replace the NodataValue of the finer resolution subgrids
        with `src` values over the overlapping region.

        Parameters
        -----------
        src: int
            index of the leaky subgrid
        """
        ndv = self.gtg_metadata["band_no_data"][0]
        for i in range(src+1, len(self.subgrid_arrays)):
            overlap = self.intersection(src, i)
            if not overlap:
                continue
            src_overlap_indices = self._index_range(g_ind=src, sub_extent=overlap)
            tgt_overlap_indices = self._index_range(g_ind=i, sub_extent=overlap)
            src_slice = deepcopy(self.subgrid_arrays[src][src_overlap_indices[1]:src_overlap_indices[3],
                                                          src_overlap_indices[0]:src_overlap_indices[2]])
            tgt_slice = deepcopy(self.subgrid_arrays[i][tgt_overlap_indices[1]:tgt_overlap_indices[3],
                                                        tgt_overlap_indices[0]:tgt_overlap_indices[2]])
            assert tgt_slice.shape >= src_slice.shape, ("subgrid resolutions must progressively increase.")
            tgt_mask = tgt_slice == ndv
            if np.count_nonzero(tgt_mask) == 0:
                continue
            src_slice[src_slice == ndv] = np.nan
            tgt_slice[tgt_mask] = np.nan
            src_slice = ski.transform.resize(src_slice, tgt_slice.shape)
            tgt_slice[tgt_mask] = src_slice[tgt_mask]
            tgt_slice = np.nan_to_num(tgt_slice, nan=ndv)
            self.subgrid_arrays[i][tgt_overlap_indices[1]:tgt_overlap_indices[3],
                                   tgt_overlap_indices[0]:tgt_overlap_indices[2]] = tgt_slice
        return

    def save_mutated_gtg(self):
        """
        Save a new GTG file with the mutated subgrid arrays.
        """        
        shutil.copy2(self.input_file, self.output_file)

        for i in range(1, self.n_subgrids+1):
            ds = gdal.Open(f"GTIFF_DIR:{i}:{self.output_file}", gdal.GA_Update)
            band = ds.GetRasterBand(1)
            band.WriteArray(self.subgrid_arrays[i-1])
            band, ds = None, None
        return

    def transform(self, output_file: str, reinitialize=False):
        self.output_file = output_file
        if reinitialize:
            self.compile_subgrid_arrays()
        for i in range(len(self.subgrid_arrays)-1):
            self.leak(src=i)
        self.save_mutated_gtg()
        return


if __name__ == "__main__":
    input_file = "us_noaa_nos_ITRF2014_LMSL_MSL_(XGEOID20B_CONUSPAC)_(DEdelches02_vdatum_4.4_20220315_1983-2001).tif"
    DiffusiveGTG(input_file=input_file).transform(output_file=input_file + ".diffused.tif")
