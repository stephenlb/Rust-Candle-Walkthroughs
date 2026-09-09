use csv;
use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};

///
/// K-means Clustering
///

fn cdist(x1: &Tensor, x2: &Tensor) -> Result<Tensor> {
    let x1 = x1.unsqueeze(0)?;
    let x2 = x2.unsqueeze(0)?;
    let out = x1
        .broadcast_sub(&x2)? // distance
        .sqr()? // positive value dist
        .sum(D::Minus1)? // sum innermost dim
        .sqrt()?
        .transpose(D::Minus1, D::Minus2)?;

    Ok(out)
}

fn load_dataset(file: &str, device: &Device) -> Result<Tensor> {
    let mut reader = csv::Reader::from_path(file)?;
    let mut data = vec![];

    for result in reader.records() {
        let record = result?;
        let mut row: Vec<f64> = vec![];
        for i in 1..5 {
            row.push(record[i].parse()?);
        }
        data.push(row);
    }

    let features = data[0].len();
    let samples = data.len();
    let data: Vec<f64> = data
        .into_iter()
        .flatten()
        .collect();
    let out = Tensor::from_slice(
        data.as_slice(),
        (samples, features),
        device
    )?;

    Ok(out)
}

fn main() -> Result<()> {
    Ok(())
}
