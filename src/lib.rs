use numpy::ndarray::{Array2, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{pymodule, types::PyModule, PyResult, Python, Bound};

#[pymodule]
fn fdcomp<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {

    fn d8comp(dir1: ArrayView2<'_, u8>, dir2: ArrayView2<'_, u8>, 
              aff1: ArrayView1<'_, f64>, aff2: ArrayView1<'_, f64>) -> Array2<u8> {

        &dir1 + &dir2
    }

    // wrapper of `axpy`
    #[pyfn(m)]
    #[pyo3(name = "d8comp")]
    fn d8comp_py<'py>(
        py: Python<'py>,
        dir1: PyReadonlyArray2<'py, u8>,
        dir2: PyReadonlyArray2<'py, u8>,
        aff1: PyReadonlyArray1<'py, f64>,
        aff2: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyArray2<u8>> {
        let dir1 = dir1.as_array();
        let dir2 = dir2.as_array();
        let aff1 = aff1.as_array();
        let aff2 = aff2.as_array();
        let res = d8comp(dir1, dir2, aff1, aff2);
        res.into_pyarray_bound(py)
    }

    Ok(())
}