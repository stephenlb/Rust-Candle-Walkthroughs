use anyhow::Result;
use candle_core::{Device, Tensor};

struct LinearRegression {
    weights: Tensor,
    bias: Tensor,
    device: Device,
}

/*
squeeze
[1,2,3] -> [[1,2,3]]
[1,2,3] -> [[1],[2],[3]]

unsqeeze
[[1,2,3]] -> [1,2,3]
*/

impl LinearRegression {
    fn new(feature: usize, device: Device) -> Result<Self> {
        // TODO fix input weights matrix input dim
        let weights: Tensor = Tensor::randn(0f32, 1f32, (2,1,3), &device)?;
        let bias: Tensor = Tensor::new(0f32, &device)?;

        Ok(LinearRegression {
            weights,
            bias,
            device,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = x.matmul(&self.weights.unsqueeze(0)?)?;
        let out = out.squeeze(1)?;
        let out = out.broadcast_add(&self.bias)?;
        Ok(out)
    }
}

fn main() -> Result<()> {
    println!("{:?}", -1);
    let device: Device = Device::metal_if_available(0)?;
    let model: LinearRegression = LinearRegression::new(10, device.clone())?;
    let x: Tensor = Tensor::randn(0f32, 1f32, (10, 1), &device)?;
    //model.forward(&x)?;
    //println!("working");
    Ok(())
}

