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

fn k_means(data: &Tensor, clusters: usize, max_iterations: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let (items, _) = data.dims2()?;
    let mut rng = rand::rng();
    let mut indicies: Vec<_> = (0..items).collect();
    indicies.shuffle(&mut rng);
    let centeroid_idx: Vec<_> = indicies[..clusters]
        .iter()
        .copied()
        .map(|x| x as i64 )
        .collect();
    let centroid_idx_tensor = Tensor::from_slice(
        centeroid_idx.as_slice(), 
        (clusters,),
        device,
    )?;

    let mut centers = data.index_select(&centroid_idx_tensor, 0)?;
    let mut cluster_assignments = Tensor::zeros(
        (items,),
        DType::U32,
        device,
    )?;

    for _ in 0..max_iterations {
        let dist: Tensor = cdist(data, &centers)?;
        cluster_assignments = dist.argmin(D::Minus1)?;
        let mut centers_vec = vec![];
        //let mut centers_vec: Vec<f32> = Vec::new();

        for c in 0..clusters {
            let mut indicies: Vec<u32> = vec![];
            cluster_assignments
                .to_vec1::<u32>()?
                .iter()
                .enumerate()
                .for_each(|(j, x)| {
                    if *x == c as u32 {
                        indicies.push(j as u32);
                    }
                });
            let indicies = Tensor::from_slice(
                indicies.as_slice(),
                (indicies.len(),),
                device,
            )?;
            let cluster_data = data.index_select(&indicies, 0)?;
            let average = cluster_data.mean(0)?;
            centers_vec.push(average);

        }
        centers = Tensor::stack(centers_vec.as_slice(), 0)?;
    }

    Ok((centers, cluster_assignments))
}

fn main() -> Result<()> {
    let file: &str = "data/iris.csv";
    let device: Device = Device::Cpu;
    let data: Tensor = load_dataset(
        file,
        &device,
    )?;

    let (centers, cluster_assignments) =
        k_means(
            &data,
            5 as usize,
            20 as usize,
            &device,
        )?;

    println!("{}", centers);
    println!("{}", cluster_assignments);


    println!("data: {data}");
    Ok(())
}
