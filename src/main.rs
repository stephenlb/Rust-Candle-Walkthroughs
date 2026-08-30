use anyhow::Result;
use candle_core::{Device, Tensor};

fn load_data(device: &Device) -> Result<Tensor> {
    let data: &[f32] = &[
        1.0, 1.0, 1.2, 1.6, 1.1, 1.8, 2.0,
        1.0, 1.0, 1.2, 1.6, 1.1, 1.8, 2.0,
        1.0, 1.0, 1.2, 700.6, 1.1, 1.8, 2.0,
        1.0, 1.0, 1.2, 1.6, 1.1, 1.8, 2.0,
    ];
    let tensor: Tensor = Tensor::from_slice(data, (4,7), device)?;

    Ok(tensor)
}

trait TensorExt {
    fn stephen_norm(&self) -> Result<Tensor>;
}

impl TensorExt for Tensor {
    fn stephen_norm(&self/*, data: &Tensor*/) -> Result<Tensor> {
        let average = self.mean(0)?;
        let t: Tensor = Tensor::from_slice(&[1.0f32], (1,), self.device())?;
        //println!("average: {average}");
        let diff = self.broadcast_sub(&average)?.sqr()?.broadcast_add(&t)?;
        ////println!("diff: {diff}");
        let variance = diff.mean(0)?;
        //println!("variance: {variance}");
        let standard_deviation = variance.sqrt()?;
        //println!("standard_deviation: {standard_deviation}");
        let norm = self.broadcast_sub(&average)?.broadcast_div(&standard_deviation)?;
        Ok(norm)
    }
}

fn stephen_p_x(
    data: &Tensor,
    mean: &Tensor,
    two_variance: &Tensor,
    two_pi_sqrt_std_dev: &Tensor,
    device: &Device,
) -> Result<Vec<f32>> {
    let px = data
        .broadcast_sub(mean)?
        .sqr()?
        .broadcast_div(&two_variance)?
        .exp()?
        .broadcast_mul(two_pi_sqrt_std_dev)?
        //.broadcast_sub(&t)?
        .recip()?;

    //println!("px {:?}", px);
    let pax: Vec<f32> = px.to_vec1()?;
    Ok(pax)
    /*
    let px: f32 = pax
        .into_iter()
        .fold(1.0f32, |acc, x| acc * x);
    */
    //Ok(px)
}

const THRESHOLD: f32 = 0.1;

fn main() -> Result<()> {
    //println!("Hello, world!");
    let device: Device = Device::metal_if_available(0)?;
    let data: Tensor = load_data(&device)?;
    let norm: Tensor = data.stephen_norm()?;
    //println!("Norm: {norm}");

    let mean = norm.mean(0)?;
    let variance = norm.broadcast_sub(&mean)?.sqr()?.mean(0)?;
    let std_dev = variance.sqrt()?;

    let two_variance = variance.broadcast_mul(&Tensor::new(2.0f32, &device)?)?;
    let pi2: Tensor = Tensor::new((std::f64::consts::PI * 2.0) as f32, &device)?;
    let two_pi_sqrt_std_dev = std_dev.broadcast_mul(&pi2)?.sqrt()?;

    let rows = norm.shape().dims2()?.0;
    let mut anomalies = 0;
    for row in 0..rows {
        //println!("{row}");
        let selector = Tensor::new(&[row as u32],  &device)?;
        //println!("norm: {norm}");
        let row_tensor = norm
            .index_select(&selector, 0)?
            .squeeze(0)?;
        //println!("row_tenso//r: {row_tensor}");
        let px: Vec<f32> = stephen_p_x(
            &row_tensor,
            &mean,
            &two_variance,
            &two_pi_sqrt_std_dev,
            &device,
        )?;
        for (col, item) in px.iter().enumerate() {
            println!("item: {item}");
            /*
            if item.is_nan() {
                continue;
            }*/
            if (item < &THRESHOLD) {
                anomalies += 1;
            }
        }
        //println!("row_tensor: {row_tensor}");
        println!("px: {:?}", px);
    }

    println!("Anomalies: {anomalies}");

    Ok(())
}
