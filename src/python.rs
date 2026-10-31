//! Python bindings. Everything here translates between `NumPy` and the pure Rust core.

#![expect(
    clippy::needless_pass_by_value,
    reason = "PyO3 extracts function arguments by value"
)]

use std::collections::HashMap;

use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;

use crate::{Config, Detection, LabelId, Motion, Signature, Tracker};

const CHANNELS: usize = 3;
const SIGNATURE_LENGTH: usize = 6;

/// Reads an `N×M` float array into fixed-size rows.
fn rows<const M: usize>(array: &PyReadonlyArray2<'_, f64>, name: &str) -> PyResult<Vec<[f64; M]>> {
    let view = array.as_array();
    if view.ncols() != M {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (N, {M})"
        )));
    }
    Ok(view
        .rows()
        .into_iter()
        .map(|row| std::array::from_fn(|column| row[column]))
        .collect())
}

/// Computes signatures for many boxes in one call, without holding the GIL.
#[pyfunction]
fn rgb_mean_std<'py>(
    py: Python<'py>,
    pixels: PyReadonlyArray3<'py, u8>,
    boxes: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let &[height, width, channels] = pixels.shape() else {
        return Err(PyValueError::new_err("pixels must have shape (H, W, 3)"));
    };
    if channels != CHANNELS {
        return Err(PyValueError::new_err("pixels must have shape (H, W, 3)"));
    }
    if !pixels.is_c_contiguous() {
        return Err(PyValueError::new_err("pixels must be C-contiguous"));
    }
    let data = pixels.as_slice()?;
    let boxes = rows::<4>(&boxes, "boxes")?;
    let stride = width * CHANNELS;
    let flat: Vec<f64> = py.detach(|| {
        boxes
            .iter()
            .flat_map(|&bbox| crate::rgb_mean_std(data, width, height, stride, bbox))
            .collect()
    });
    let signatures = Array2::from_shape_vec((boxes.len(), SIGNATURE_LENGTH), flat)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(signatures.into_pyarray(py))
}

/// Tuned tracker values.
#[pyclass(name = "Config", frozen)]
struct PyConfig(Config);

#[pymethods]
impl PyConfig {
    /// The values tuned for Age of Empires II.
    #[staticmethod]
    fn aoe2() -> Self {
        Self(Config::aoe2())
    }
}

/// `"stable"`, `"reset"` or `("pan", dx, dy)`.
fn parse_motion(motion: &Bound<'_, PyAny>) -> PyResult<Motion> {
    if let Ok(kind) = motion.cast::<PyString>() {
        return match kind.to_str()? {
            "stable" => Ok(Motion::Stable),
            "reset" => Ok(Motion::Reset),
            other => Err(PyValueError::new_err(format!("unknown motion: {other:?}"))),
        };
    }
    let (kind, dx, dy): (String, f64, f64) = motion.extract()?;
    if kind == "pan" {
        Ok(Motion::Pan { dx, dy })
    } else {
        Err(PyValueError::new_err(format!("unknown motion: {kind:?}")))
    }
}

/// `(track_ids, labels, agreements)`, with `NaN` where there is no agreement yet.
type Observation<'py> = (
    Bound<'py, PyArray1<u64>>,
    Vec<Py<PyString>>,
    Bound<'py, PyArray1<f64>>,
);

/// Links detections across frames. Keep one per video stream.
#[pyclass(name = "Tracker")]
struct PyTracker {
    inner: Tracker,
    label_ids: HashMap<String, LabelId>,
    label_names: Vec<Py<PyString>>,
}

impl PyTracker {
    fn intern(&mut self, label: &Bound<'_, PyString>) -> PyResult<LabelId> {
        let text = label.to_str()?;
        if let Some(&id) = self.label_ids.get(text) {
            return Ok(id);
        }
        let id = LabelId::try_from(self.label_names.len())
            .map_err(|_| PyValueError::new_err("too many distinct labels"))?;
        self.label_ids.insert(text.to_owned(), id);
        self.label_names.push(label.clone().unbind());
        Ok(id)
    }

    fn label_name(&self, py: Python<'_>, id: LabelId) -> PyResult<Py<PyString>> {
        usize::try_from(id)
            .ok()
            .and_then(|index| self.label_names.get(index))
            .map(|name| name.clone_ref(py))
            .ok_or_else(|| PyValueError::new_err("unknown label ID"))
    }
}

#[pymethods]
impl PyTracker {
    #[new]
    fn new(config: PyRef<'_, PyConfig>) -> Self {
        Self {
            inner: Tracker::new(config.0),
            label_ids: HashMap::new(),
            label_names: Vec::new(),
        }
    }

    /// Whether a frame is newer than the last one by both ID and time.
    fn accepts(&self, capture_id: i64, captured_at: f64) -> bool {
        self.inner.accepts(capture_id, captured_at)
    }

    /// Drops every track. The ID counter and the ordering guard are kept.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Processes one frame. Returns `None` if the frame is not newer than the last.
    #[pyo3(signature = (*, capture_id, captured_at, motion, boxes, labels, confidences, signatures))]
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the keyword-only Python API"
    )]
    fn observe<'py>(
        &mut self,
        py: Python<'py>,
        capture_id: i64,
        captured_at: f64,
        motion: &Bound<'py, PyAny>,
        boxes: PyReadonlyArray2<'py, f64>,
        labels: Vec<Bound<'py, PyString>>,
        confidences: PyReadonlyArray1<'py, f64>,
        signatures: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Option<Observation<'py>>> {
        let motion = parse_motion(motion)?;
        let boxes = rows::<4>(&boxes, "boxes")?;
        let signatures: Vec<Signature> = rows::<SIGNATURE_LENGTH>(&signatures, "signatures")?;
        let confidences = confidences.as_array();
        let count = boxes.len();
        if labels.len() != count || confidences.len() != count || signatures.len() != count {
            return Err(PyValueError::new_err(
                "boxes, labels, confidences and signatures must have the same length",
            ));
        }
        let mut detections = Vec::with_capacity(count);
        for index in 0..count {
            detections.push(Detection {
                bbox: boxes[index],
                label: self.intern(&labels[index])?,
                confidence: confidences[index],
                signature: signatures[index],
            });
        }

        let Some(tracked) = self
            .inner
            .observe(capture_id, captured_at, motion, &detections)
        else {
            return Ok(None);
        };
        let track_ids: Vec<u64> = tracked.iter().map(|item| item.track_id).collect();
        let agreements: Vec<f64> = tracked
            .iter()
            .map(|item| item.agreement.unwrap_or(f64::NAN))
            .collect();
        let names = tracked
            .iter()
            .map(|item| self.label_name(py, item.label))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Some((
            PyArray1::from_vec(py, track_ids),
            names,
            PyArray1::from_vec(py, agreements),
        )))
    }

    /// Detections with two near-equal candidates in the last accepted frame.
    #[getter]
    fn ambiguous_count(&self) -> usize {
        self.inner.stats().ambiguous_count
    }

    /// Feasible track-to-detection pairs in the last accepted frame.
    #[getter]
    fn candidate_count(&self) -> usize {
        self.inner.stats().candidate_count
    }

    /// Time spent generating candidates in the last accepted frame.
    #[getter]
    fn candidate_ms(&self) -> f64 {
        self.inner.stats().candidate_ms
    }

    /// Time spent on ambiguity, matching and output in the last accepted frame.
    #[getter]
    fn matching_ms(&self) -> f64 {
        self.inner.stats().matching_ms
    }
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(rgb_mean_std, module)?)?;
    module.add_class::<PyConfig>()?;
    module.add_class::<PyTracker>()?;
    Ok(())
}
