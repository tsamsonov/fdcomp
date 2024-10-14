use numpy::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::{pymodule, types::PyModule, PyResult, Python, Bound};
use std::collections::BinaryHeap;
use std::collections::HashSet;
use std::vec::Vec;

const NODATA: u8 = 255;
const RADIUS: f64 = 6371000.0;

#[pymodule]
fn fdcomp<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {

    fn ij2lonlat(i: usize, j: usize, aff: ArrayView1<'_, f64>) -> (f64, f64) {
        (
            (j as f64 * aff[0] + i as f64 * aff[1] + aff[2]).to_radians(),
            (j as f64 * aff[3] + i as f64 * aff[4] + aff[5]).to_radians()
        )
    }

    fn earth_dist_vincenty(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> u32 {
        let x = lat1.sin() * lat2.sin() + lat1.cos() * lat2.cos() * (lon2 - lon1).cos();
        let y1 = (lat2.cos() * (lon2 - lon1).sin()).powf(2.0);
        let y2 = (lat1.cos() * lat2.sin() - lat1.sin() * lat2.cos() * (lon2 - lon1).cos()).powf(2.0);
        return (RADIUS * (y1 + y2).sqrt().atan2(x)) as u32
    }

    fn earth_dist(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> u32 {
        let sigma = lat1.sin() * lat2.sin() + lat1.cos() * lat2.cos() * (lon2-lon1).cos();
        return (RADIUS * sigma.acos()) as u32
    }

    // compute drainage tree
    fn d8tree(acc: ArrayView2<'_, u32>, dir: ArrayView2<'_, u8>, aff: ArrayView1<'_, f64>,) -> (Array2<u32>, Array2<u32>)  {

        let dj = [1, 1, 0, -1, -1, -1,  0,  1]; // columns
        let di = [0, 1, 1,  1,  0, -1, -1, -1]; // rows
        let mut idx: [usize; 129] = [0usize; 129];

        // This maps Esri-style D8 pointer values
        // onto the cell offsets in d_x and d_y.
        idx[1] = 0usize;
        idx[2] = 1usize;
        idx[4] = 2usize;
        idx[8] = 3usize;
        idx[16] = 4usize;
        idx[32] = 5usize;
        idx[64] = 6usize;
        idx[128] = 7usize;

        let shape = dir.raw_dim();
        let mut res = Array2::zeros(shape);

        let mut seeds: Vec<Vec<u32>> = Vec::new();

        let nrow = shape[0];
        let ncol = shape[1];

        let mut queue = BinaryHeap::new();

        for i in 0..nrow {
            for j in 0..ncol {
                if dir[[i, j]] <= 128 {
                    queue.push((acc[[i, j]], i, j));
                }
            }
        }

        let mut id = 0_u32;
        let mut amax = 0_u32;
        let (mut i, mut j): (usize, usize);
        let (mut ik, mut jl): (isize, isize);
        let (mut uik, mut ujl, mut imax, mut jmax): (usize, usize, usize, usize);
        while queue.len() > 0 {
            let (a, mut istart, mut jstart) = queue.pop().unwrap();
            if res[[istart, jstart]] == 0 {
                id += 1;
                i = istart;
                j = jstart;
                res[[i, j]] = id;

                let mut length = 0_u32;
                let mut ncells = 1_u32;

                while acc[[i, j]] > 0 {
                    amax = 0;
                    imax = i;
                    jmax = j;

                    for nb in 0..8 {
                        let k = di[nb];
                        let l = dj[nb];  

                        ik = i as isize + k;
                        jl = j as isize + l;

                        if ik < 0 || ik >= nrow as isize || jl < 0 || jl >= ncol as isize {
                            continue;
                        }

                        uik = ik as usize;
                        ujl = jl as usize;

                        if dir[[uik, ujl]] <= 128 {
                            let diridx = idx[dir[[uik, ujl]] as usize];

                            if (acc[[uik, ujl]] >= amax) && (i == (ik + di[diridx]) as usize) && (j == (jl + dj[diridx]) as usize) {
                                imax = uik;
                                jmax = ujl;
                                amax = acc[[uik, ujl]]
                            }
                        }
                    }
                    
                    if (i == imax) && (j == jmax) { break; }

                    let (lon1, lat1) = ij2lonlat(i, j, aff);
                    let (lon2, lat2) = ij2lonlat(imax, jmax, aff);
                    length += earth_dist_vincenty(lon1, lat1, lon2, lat2);
                    ncells += 1;

                    i = imax;
                    j = jmax;
                    res[[i, j]] = id;

                    if dir[[i, j]] == 0 { break; }
                }
                let seed = vec![i as u32, j as u32, istart as u32, jstart as u32, ncells, a, length];
                seeds.push(seed);
            }
        }

        let mut pyseeds = Array2::zeros([seeds.len(), 7]);
        for i in 0..seeds.len() {
            for j in 0..7 {
                pyseeds[[i, j]] = seeds[i][j];
            }
        }

        return (res, pyseeds)

    }

    fn d8comp(dir1: ArrayView2<'_, u8>, dir2: ArrayView2<'_, u8>, seeds: ArrayView2<'_, u32>) -> Array1<f64> {
 
        let shape1 = dir1.raw_dim();
        let nrow1 = shape1[0];
        let ncol1 = shape1[1];

        let shape2 = dir2.raw_dim();
        let nrow2 = shape2[0];
        let ncol2 = shape2[1];

        let ratio = nrow1 / nrow2;

        let dj = [1, 1, 0, -1, -1, -1,  0,  1]; // columns
        let di = [0, 1, 1,  1,  0, -1, -1, -1]; // rows
        let mut idx: [usize; 129] = [0usize; 129];

        // This maps Esri-style D8 pointer values
        // onto the cell offsets in d_x and d_y.
        idx[1] = 0usize;
        idx[2] = 1usize;
        idx[4] = 2usize;
        idx[8] = 3usize;
        idx[16] = 4usize;
        idx[32] = 5usize;
        idx[64] = 6usize;
        idx[128] = 7usize;

        let (mut i, mut j): (usize, usize);
        let mut flag: bool;
        let mut dir: u8;

        let shape_seeds = seeds.raw_dim();
        let nseeds = shape_seeds[0];
        let mut res = Array1::zeros(nseeds);

        for k in 0..nseeds {
            let i1 = seeds[[k, 0]] as usize;
            let j1 = seeds[[k, 1]] as usize;

            let i2 = seeds[[k, 2]] as usize;
            let j2 = seeds[[k, 3]] as usize;

            // Trace the flowline from generalized cell

            let istart = i1 / ratio;
            let jstart = j1 / ratio;

            flag = false;
            i = istart;
            j = jstart;

            let mut dir2cells: HashSet<(usize, usize)> = HashSet::new();
            dir2cells.insert((i / ratio, j / ratio));

            while !flag {
                // find its downslope neighbour
                dir = dir2[[i, j]];
                if dir != NODATA && dir > 0 {
                    // move x and y accordingly
                    i = (i as isize + di[idx[dir as usize]]) as usize;
                    j = (j as isize + dj[idx[dir as usize]]) as usize;
                    dir2cells.insert((i, j));
                } else {
                    flag = true;
                }
            }
            
            i = i1;
            j = j1;

            let mut total = 1;
            let mut inter = 1;

            while i != i2 || j != j2 {
                dir = dir1[[i, j]];
                i = (i as isize + di[idx[dir as usize]]) as usize;
                j = (j as isize + dj[idx[dir as usize]]) as usize;
                total += 1;
                inter += if dir2cells.contains(&(i / ratio, j / ratio)) { 1 } else { 0 };
            }
            res[k] = inter as f64 / total as f64;

        }
        return res
    }

    // wrapper of `d8comp`
    #[pyfn(m)]
    #[pyo3(name = "d8comp")]
    fn d8comp_py<'py>(
        py: Python<'py>,
        dir1: PyReadonlyArray2<'py, u8>,
        dir2: PyReadonlyArray2<'py, u8>,
        seeds: PyReadonlyArray2<'py, u32>,
    ) -> Bound<'py, PyArray1<f64>> {
        let dir1 = dir1.as_array();
        let dir2 = dir2.as_array();
        let seeds = seeds.as_array();
        let res = d8comp(dir1, dir2, seeds);
        res.into_pyarray_bound(py)
    }

    // wrapper of `d8tree`
    #[pyfn(m)]
    #[pyo3(name = "d8tree")]
    fn d8tree_py<'py>(
        py: Python<'py>,
        acc: PyReadonlyArray2<'py, u32>,
        dir: PyReadonlyArray2<'py, u8>,
        aff: PyReadonlyArray1<'py, f64>
    ) -> (Bound<'py, PyArray2<u32>>, Bound<'py, PyArray2<u32>>) {
        let acc = acc.as_array();
        let dir = dir.as_array();
        let aff = aff.as_array();
        let (res, pyseeds) = d8tree(acc, dir, aff);
        (res.into_pyarray_bound(py), pyseeds.into_pyarray_bound(py))
    }

    Ok(())
}