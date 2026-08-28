use anyhow::Result;
use candle_core::{Device, Tensor};

fn load_data(device: &Device) -> Result<Tensor> {
    let data: &[f32] = &[
        1.0, 1.0, 1.2, 1.6, 1.1, 1.8, 2.0,
        1.0, 1.0, 1.2, 1.6, 1.1, 1.8, 2.0,
        1.0, 1.0, 1.2, 7.6, 4.1, 1.8, 2.0,
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
        println!("average: {average}");
        let diff = self.broadcast_sub(&average)?.sqr()?.broadcast_add(&t)?;
        println!("diff: {diff}");
        let varience = diff.mean(0)?;
        println!("varience: {varience}");
        let standard_deviation = varience.sqrt()?;
        println!("standard_deviation: {standard_deviation}");
        let norm = self.broadcast_sub(&average)?.broadcast_div(&standard_deviation)?;
        Ok(norm)
    }
}

fn stephen_p_x(
    data: &Tensor,
    mean: &Tensor,
    two_varience: &Tensor,
    two_pi_sqrt_std_dev: &Tensor,
) -> Result<f64> {
    let px = data
        .broadcast_sub(mean)?
        .sqr()?
        .broadcast_div(two_varience)?
        .exp()?
        .broadcast_mul(two_pi_sqrt_std_dev)?
        .recip()?;

    let pax: Vec<f64> = px.to_vec1()?;
    let px: f64 = pax
        .into_iter()
        .fold(1.0, |acc, x| acc * x);
    Ok(px)
}

fn main() -> Result<()> {
    println!("Hello, world!");
    let device: Device = Device::metal_if_available(0)?;
    let data: Tensor = load_data(&device)?;
    let norm: Tensor = data.stephen_norm()?;
    println!("Norm: {norm}");

    let mean = norm.mean(0)?;
    let varience = norm.broadcast_sub(&mean)?.sqr()?.mean(0)?;
    let std_dev = varience.sqrt()?;

    let two_varience = varience.broadcast_mul(&Tensor::new(2.0, &device)?)?;
    let pi2: Tensor = Tensor::new(std::f64::consts::PI * 2.0, &device)?;
    let two_pi_sqrt_std_dev = std_dev.broadcast_mul(&pi2)?.sqrt()?;

    let rows = data.shape().dims2()?.0;

    Ok(())
}
