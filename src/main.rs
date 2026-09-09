use csv;
use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};

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

fn main() -> Result<()> {
    Ok(())
}
