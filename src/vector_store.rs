use ndarray::{Array1, Array2};

pub struct VectorStore {
    pub data: Array1<Vector>,
}

pub struct Vector {
    pub id: u64,
    pub data: Array1<f32>,
    pub timestamp: u64,
}

impl VectorStore {
    pub fn new(vectors_data: Vec<(u64, Vec<f32>, u64)>) -> Self {
        let mut vectors = Vec::<Vector>::new();
        for vector_data in vectors_data.into_iter() {
            vectors.push(Vector {
                id: vector_data.0,
                data: Array1::from_vec(vector_data.1),
                timestamp: vector_data.2,
            });
        }
        VectorStore {
            data: Array1::from_vec(vectors),
        }
    }

    pub fn get_vectors(&self) -> Array2<f32> {
        let n = self.data.len();
        let dim = self.data[0].data.len();
        let mut vectors_arr = Array2::<f32>::default((n, dim));
        for i in 0..n {
            for j in 0..dim {
                vectors_arr[(i, j)] = self.data[i].data[j];
            }
        }
        vectors_arr
    }
}
