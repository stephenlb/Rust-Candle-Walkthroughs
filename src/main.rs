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
    fn stephen_norm(&self/*, data: &Tensor*/) -> Result<Tensor>;
}

impl TensorExt for Tensor {
    fn stephen_norm(&self/*, data: &Tensor*/) -> Result<Tensor> {
        let average = self.mean(0)?;
        //let t  = Tensor::new
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
/*fn normalize(data: &Tensor) -> Result<Tensor> {
    let mean = data.mean(0);
}*/

fn main() -> Result<()> {
    println!("Hello, world!");
    let device: Device = Device::metal_if_available(0)?;
    let data: Tensor = load_data(&device)?;
    let norm: Tensor = data.stephen_norm()?;
    println!("Norm: {norm}");

    Ok(())
}
