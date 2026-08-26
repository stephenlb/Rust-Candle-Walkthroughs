use anyhow::Result;
use candle_core::{Device, Tensor};

fn load_data(device: &Device) -> Result<Tensor> {
    let data: &[f32] = &[
        1.0, 1.0, 1.2, 1.6, 1.1, 1.8, 2.0,
        1.0, 1.0, 1.2, 7.6, 1.1, 1.8, 2.0,
        1.0, 1.0, 1.2, 1.6, 1.1, 1.8, 2.0,
    ];
    let tensor: Tensor = Tensor::from_slice(data, (3,7), device)?;

    println!("{tensor}");

    Ok(tensor)
}

trait TensorExt {
    fn norm(&self, data: &Tensor) -> Result<Tensor>;
}

impl TensorExt for Tensor {
    fn norm(&self, data: &Tensor) -> Result<Tensor> {
        Ok(Tensor::new(1.0, self.device())?)
    }
}
/*fn normalize(data: &Tensor) -> Result<Tensor> {
    let mean = data.mean(0);
}*/

fn main() -> Result<()> {
    println!("Hello, world!");
    let device: Device = Device::metal_if_available(0)?;
    let data: Tensor = load_data(&device)?;

    Ok(())
}
