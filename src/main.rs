use anyhow::Result;
use candle_core::{Device, Tensor, D};
use nalgebra::DMatrix;

fn invert_tensor(tensor: &Tensor) -> Result<Tensor> {
    let (rows, cols) = tensor.dims2()?;
    let data = tensor.flatten_all()?.to_vec1::<f32>()?;

    // Invert using nalgebra
    let matrix = DMatrix::from_row_slice(rows, cols, &data);
    let inv_matrix = matrix.try_inverse()
        .ok_or_else(|| candle_core::Error::Msg("Matrix is singular".to_string()))?;

    // Convert back to Candle
    Ok(Tensor::from_slice(inv_matrix.as_slice(), (rows, cols), tensor.device())?)
}

struct LinearRegression {
    weights: Tensor,
    bias: Tensor,
    device: Device,
    lr: f32,
    regularization: f32,
}

impl LinearRegression {
    fn new(features: usize, device: Device) -> Result<Self> {
        // TODO fix input weights matrix input dim
        let weights: Tensor = Tensor::randn(0f32, 1f32, (features, 1), &device)?;
        let bias: Tensor = Tensor::randn(0f32, 1f32, (1), &device)?;
        let lr: f32 = 0.003;
        let regularization: f32 = 0.01;

        Ok(LinearRegression {
            weights,
            bias,
            device,
            lr,
            regularization,
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

    fn train(
        &mut self, 
        features: &Tensor,
        labels: &Tensor,
    ) -> Result<()> {
        let (batch, _) = features.shape().dims2()?;
        let reg = self.regularization;
        let lr = self.lr;
        let output = self.forward(features)?;
        let loss = output.sub(labels)?;
        let reg = Tensor::new(self.regularization / batch as f32, &self.device)?;
        let regularization = self
            .weights
            .broadcast_mul(&reg)?;

        /*let gradient = features
            .t()?
            .matmul(&loss.unsqueeze(D::Minus1)?)?;
        
        println!("gradient: {gradient}");
        */
        println!("regularization: {regularization}");
        //println!("loss: {loss}");
        //println!("output: {output}");

        Ok(())
    }

    ///@Lenam: derivative tell u if increase 
    /// weight would it increase
    /// loss or not.
    /// then decide to update weight to reduce loss

    /*
    here is numpy code to do the thing
    xm = x.mean()
    ym = y.mean()
    X = x-xm
    Y = y-ym
    W = np.linalg.inv(X.transpose() @ X).inverse() @ X @ Y

    def pred(x):
        return W @ (x-xm)+ym + b
    */

    /*
    xm = x.mean()
    ym = y.mean()
    X = x-x.mean()
    Y = y-y.mean()
    W = np.linalg.inv(X.transpose() @ X).inverse() @ X @ Y
    biass = ym-W@xmdef

    pred(x):
        return W@x+biass
    */

    fn fit(
        &mut self,
        features: &Tensor,
        labels: &Tensor,
    ) -> Result<f32> {
        let labels = if labels.rank() == 1 {
            labels.unsqueeze(D::Minus1)?
        } else {
            labels.clone()
        };

        // Center each column on its own mean, not one global mean.
        let xm = features.mean(0)?; // (features,)
        let ym = labels.mean(0)?;   // (1,)
        let x = features.broadcast_sub(&xm)?; // (samples, features)
        let y = labels.broadcast_sub(&ym)?;  // (samples, 1)

        // W = inv(X.t X) X.t Y
        let xt = x.t()?.contiguous()?;
        let xtx = xt.matmul(&x)?;
        let xtx_inv = invert_tensor(&xtx)?;
        let weights = xtx_inv.matmul(&xt.matmul(&y)?)?; // (features, 1)

        // bias = ym - xm · W
        let bias = ym.sub(&xm.unsqueeze(0)?.matmul(&weights)?.squeeze(0)?)?;

        self.weights = weights;
        self.bias = bias;

        let predictions = self.forward(features)?;
        self.loss(&predictions, &labels)
    }
}

fn main() -> Result<()> {
    let device: Device = Device::metal_if_available(0)?;
    let mut model: LinearRegression = LinearRegression::new(10, device.clone())?;
    let features: Tensor = Tensor::randn(0f32, 1f32, (200, 10), &device)?;
    let labels: Tensor = Tensor::randn(0f32, 1f32, (10, 1), &device)?;
    let labels: Tensor = features.matmul(&labels)?.affine(1.0, 5.0)?;
    //println!("{features}");
    //println!("{labels}");

    //model.train(&features, &labels)?;
    let loss = model.fit(&features, &labels)?;
    println!("Loss: {loss}");

    Ok(())
}

