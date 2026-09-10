use csv;
use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use rand::prelude::*;

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

fn k_means(data: &Tensor, cluters: usize, device: &Device) -> Result<()> {
    let (n, _) = data.dims2()?;
    //let mut rng = rand::thread_rng();
    let mut rng = rand::rng().random_range(1..=100);
    Ok(())
}

fn main() -> Result<()> {
    let file: &str = "data/iris.csv";
    let device: Device = Device::metal_if_available(0)?;
    let data: Tensor = load_dataset(
        file,
        &device,
    )?;
    println!("data: {data}");
    Ok(())
}
