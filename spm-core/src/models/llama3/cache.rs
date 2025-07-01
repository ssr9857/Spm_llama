use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor, D};

use super::{Config, MAX_SEQ_LEN};

/// Abstraction over cosine and sine tables, kv-caching and attention masking.
#[derive(Debug, Clone)]
pub struct Cache {
    cos: Tensor,
    sin: Tensor,

    masks: HashMap<usize, Tensor>,
    use_kv_cache: bool,
    kvs: Vec<Option<(Tensor, Tensor)>>,

    device: Device,
}

impl Cache {
    /// Creates a new cache instance with the provided configuration.
    /// Set `use_kv_cache` to false to disable kv-caching.
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        // precompute freqs_cis
        let n_elem = config.hidden_size / config.num_attention_heads;

        log::debug!("cache::n_elem = {n_elem}");

        let theta: Vec<_> = (0..n_elem)
            .step_by(2)
            .map(|i| 1f32 / config.rope_theta.powf(i as f32 / n_elem as f32))
            .collect();

        let theta = Tensor::new(theta.as_slice(), device)?;

        log::debug!("cache::theta = {}", &theta);

        let idx_theta = Tensor::arange(0, super::MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((super::MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;

        log::debug!("cache::idx_theta = {}", &idx_theta);

        // This is different from the paper, see:
        // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;

        log::debug!("cache::cos = {}", &cos);
        log::debug!("cache::sin = {}", &sin);

        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; config.num_hidden_layers],
            device: device.clone(),
            cos,
            sin,
        })
    }

    /// Return true if kv-caching is enabled.
    pub fn with_kv_cache(&self) -> bool {
        self.use_kv_cache
    }

    /// Return the cached cosine value for the given position and sequence length.
    pub fn cosine(&self, index_pos: usize, seq_len: usize) -> Result<Tensor> {
        self.cos.narrow(0, index_pos, seq_len)
    }

    /// Return the cached sine value for the given position and sequence length.
    pub fn sine(&self, index_pos: usize, seq_len: usize) -> Result<Tensor> {
        self.sin.narrow(0, index_pos, seq_len)
    }

    /// Get the attention mask for the given sequence length.
    pub fn mask(&mut self, seq_len: usize) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&seq_len) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (seq_len, seq_len), &self.device)?;
            self.masks.insert(seq_len, mask.clone());
            Ok(mask)
        }
    }

    /// Process the input k and v by either generating their cache entry or applying a previously cached one.
    pub fn process_kv(
        &mut self,
        block_idx: usize,
        mut k: Tensor,
        mut v: Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if self.use_kv_cache {
            // if this block_idx in cache
            if let Some((cache_k, cache_v)) = &self.kvs[block_idx] {
                // update cache entry
                k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;
                let k_seq_len = k.dims()[1];
                if k_seq_len > MAX_SEQ_LEN {
                    k = k
                        .narrow(D::Minus1, k_seq_len - MAX_SEQ_LEN, MAX_SEQ_LEN)?
                        .contiguous()?
                }
                let v_seq_len = v.dims()[1];
                if v_seq_len > 2 * MAX_SEQ_LEN {
                    v = v
                        .narrow(D::Minus1, v_seq_len - MAX_SEQ_LEN, MAX_SEQ_LEN)?
                        .contiguous()?
                }
            }
            // set entry for this block
            self.kvs[block_idx] = Some((k.clone(), v.clone()))
        }
        Ok((k, v))
    }

    /// Return a copy of this cache with the same state but new kv table.
    pub fn as_new(&self) -> Self {
        let mut copy = self.clone();
        copy.clear();
        copy
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.masks.clear();
        self.kvs = vec![None; self.kvs.len()];
    }
}
