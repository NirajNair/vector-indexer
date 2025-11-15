use ndarray::Array2;

pub struct VectorStore {
    pub data: Array2<f32>,
}

impl VectorStore {
    pub fn new_vector_store(vectors: Vec<Vec<f32>>) -> Self {
        let n = vectors.len();
        let dim = vectors[0].len();
        let mut arr = Array2::<f32>::zeros((n, dim));
        for (i, vector) in vectors.into_iter().enumerate() {
            for (j, val) in vector.into_iter().enumerate() {
                arr[(i, j)] = val;
            }
        }
        VectorStore { data: arr }
    }
}
