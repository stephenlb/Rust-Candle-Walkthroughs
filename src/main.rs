use anyhow::Result;
use candle_core::{Device, Tensor};

struct LinearRegression {
    weights: Tensor,
    bias: Tensor,
    device: Device,
}

impl LinearRegression {
    fn new(feature: usize, device: Device) -> Result<Self> {
        // TODO fix input weights matrix input dim
        let weights: Tensor = Tensor::randn(0f32, 1f32, (10, 1), &device)?;
        let bias: Tensor = Tensor::randn(0f32, 1f32, (1), &device)?;

        Ok(LinearRegression {
            weights,
            bias,
            device,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = x.matmul(&self.weights)?;
        let out = out.broadcast_add(&self.bias)?;
        Ok(out)
    }

    fn loss(&self, predictions: &Tensor, targets: &Tensor) -> Result<f32> {
        let loss = predictions.sub(targets)?;
        let loss = loss.sqr()?.mean_all()?;
        let output: f32 = loss.to_scalar()?;
        Ok(output)
    }
}

fn main() -> Result<()> {
    let device: Device = Device::metal_if_available(0)?;
    let model: LinearRegression = LinearRegression::new(10, device.clone())?;
    let x: Tensor = Tensor::randn(0f32, 1f32, (1, 10), &device)?;
    let predictions = model.forward(&x)?;
    println!("predictions: {predictions}");
    Ok(())
}

