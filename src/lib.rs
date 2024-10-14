use numpy::ndarray::{Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::{pymodule, types::PyModule, PyResult, Python, Bound};
use std::collections::BinaryHeap;
use std::vec::Vec;

const NODATA: u8 = 255;

#[pymodule]
fn fdcomp<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {

    // compute drainage tree
    fn d8tree(acc: ArrayView2<'_, u32>, dir: ArrayView2<'_, u8>) -> (Array2<u32>, Array2<u32>)  {

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

        println!("Queue initialized");

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

                    i = imax;
                    j = jmax;
                    res[[i, j]] = id;

                    if dir[[i, j]] == 0 { break; }
                }
                let seed = vec![i as u32, j as u32, istart as u32, jstart as u32, a];
                seeds.push(seed);
            }
        }

        let mut pyseeds = Array2::zeros([seeds.len(), 5]);
        for i in 0..seeds.len() {
            for j in 0..5 {
                pyseeds[[i, j]] = seeds[i][j];
            }
        }

        return (res, pyseeds)

    }

    fn d8comp(acc1: ArrayView2<'_, u32>, dir1: ArrayView2<'_, u8>, dir2: ArrayView2<'_, u8>,
              aff1: ArrayView1<'_, f64>, aff2: ArrayView1<'_, f64>) -> Array2<u8> {

        let shape1 = dir1.raw_dim();
        let res = Array2::zeros(shape1);

        let nrow1 = shape1[0];
        let ncol1 = shape1[1];


        let dx = [1, 1, 1, 0, -1, -1, -1, 0];
        let dy = [-1, 0, 1, 1, 1, 0, -1, -1];
        let mut pntr_matches: [usize; 129] = [0usize; 129];

        // This maps Esri-style D8 pointer values
        // onto the cell offsets in d_x and d_y.
        pntr_matches[1] = 1usize;
        pntr_matches[2] = 2usize;
        pntr_matches[4] = 3usize;
        pntr_matches[8] = 4usize;
        pntr_matches[16] = 5usize;
        pntr_matches[32] = 6usize;
        pntr_matches[64] = 7usize;
        pntr_matches[128] = 0usize;

        let (mut x, mut y): (usize, usize);
        let mut flag: bool;
        let mut dir: u8;

        // for row in 0..nrow1 {
        //     for col in 0..ncol1 {
        //         if dir1[[row, col]] != NODATA {
        //             flag = false;
        //             x = col;
        //             y = row;
        //             while !flag {
        //                 // find its downslope neighbour
        //                 dir = dir1[[y, x]];
        //                 if dir != NODATA && dir > 0 {
        //                     // move x and y accordingly
        //                     x += dx[pntr_matches[dir]];
        //                     y += dy[pntr_matches[dir]];
        //                 } else {
        //                     flag = true;
        //                 }
        //             }
        //         }
        //     }
        // }

        return res
    }

    // wrapper of `d8comp`
    #[pyfn(m)]
    #[pyo3(name = "d8comp")]
    fn d8comp_py<'py>(
        py: Python<'py>,
        acc1: PyReadonlyArray2<'py, u32>,
        dir1: PyReadonlyArray2<'py, u8>,
        dir2: PyReadonlyArray2<'py, u8>,
        aff1: PyReadonlyArray1<'py, f64>,
        aff2: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyArray2<u8>> {
        let acc1 = acc1.as_array();
        let dir1 = dir1.as_array();
        let dir2 = dir2.as_array();
        let aff1 = aff1.as_array();
        let aff2 = aff2.as_array();
        let res = d8comp(acc1, dir1, dir2, aff1, aff2);
        res.into_pyarray_bound(py)
    }

    // wrapper of `d8tree`
    #[pyfn(m)]
    #[pyo3(name = "d8tree")]
    fn d8tree_py<'py>(
        py: Python<'py>,
        acc: PyReadonlyArray2<'py, u32>,
        dir: PyReadonlyArray2<'py, u8>
    ) -> (Bound<'py, PyArray2<u32>>, Bound<'py, PyArray2<u32>>) {
        println!("HELL YELL");
        let acc = acc.as_array();
        let dir = dir.as_array();
        let (res, pyseeds) = d8tree(acc, dir);
        (res.into_pyarray_bound(py), pyseeds.into_pyarray_bound(py))
    }

    Ok(())
}