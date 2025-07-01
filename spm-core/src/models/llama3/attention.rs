//! Causal self attention implementation.
use candle_core::{DType, Result, Tensor, D};
use candle_nn::{linear_no_bias as linear, Linear, Module, VarBuilder};


#[derive(Debug, Clone)]
pub struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
}

#[inline]
fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

impl CausalSelfAttention {
    fn apply_rotary_emb(
        &self,
        x: &Tensor,
        index_pos: usize,
        cache: &super::Cache,
    ) -> Result<Tensor> {
        let (_batch_size, _, seq_len, _hidden_size) = x.dims4()?;
        let cos = cache.cosine(index_pos, seq_len)?;
        let sin = cache.sine(index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }

    /// Process the input tensor using the given state indexes and cache.
    pub fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut super::Cache,
    ) -> anyhow::Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3().map_err(|e| anyhow!("x.dims3 -> {e}"))?;
        // 修改的时候别忘记了重新编译，不然跟二笔似的
        // log::info!("Batch size (b_sz): {}", b_sz);
        // log::info!("Sequence length (seq_len): {}", seq_len);
        // log::info!("Hidden size: {}", hidden_size);

        // log::info!("x.dims3 = {:?}", x.dims3().unwrap());

        let q = self
            .q_proj
            .forward(x)
            .map_err(|e| anyhow!("q.forward -> {e}"))?;
        let k = self
            .k_proj
            .forward(x)
            .map_err(|e| anyhow!("k.forward -> {e}"))?;
        let v = self
            .v_proj
            .forward(x)
            .map_err(|e| anyhow!("v.forward -> {e}"))?;
        // log::info!("Shape of q and k  v after apply_rotary_emb: {:?} {:?} {:?}", q.shape(), k.shape(), v.shape());


        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
            .map_err(|e| anyhow!("q.reshape -> {e}"))?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
            .map_err(|e| anyhow!("k.reshape -> {e}"))?;
        let v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)
            .map_err(|e| anyhow!("v.reshape -> {e}"))?;
        

        let q = self
            .apply_rotary_emb(&q, index_pos, cache)
            .map_err(|e| anyhow!("q.apply_rotary_emb -> {e}"))?;
        let k = self
            .apply_rotary_emb(&k, index_pos, cache)
            .map_err(|e| anyhow!("k.apply_rotary_emb -> {e}"))?;
        // log::info!("Shape of q and k v after apply_rotary_emb: {:?} {:?} {:?}", q.shape(), k.shape(), v.shape());
        

        let (k, v) = cache
            .process_kv(block_idx, k, v) // 使用kv缓存
            .map_err(|e| anyhow!("cache.process_kv(block={block_idx}) -> {e}"))?;
        // log::info!("Shape of q and k  v after process_kv: {:?} {:?} {:?}", q.shape(), k.shape(), v.shape());
        

        let k = self
            .repeat_kv(k)
            .map_err(|e| anyhow!("repeat_kv(k) -> {e}"))?;
        let v = self
            .repeat_kv(v)
            .map_err(|e| anyhow!("repeat_kv(v) -> {e}"))?;
        // log::info!("Shape of q and k  v after repeat_kv: {:?} {:?} {:?}", q.shape(), k.shape(), v.shape());

        let y = {
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            // 因为使用了kv cache, 所以seq_len大概率不为1
            let att = if seq_len == 1 {
                att
            } else {
                let mask = cache
                    .mask(seq_len)
                    .map_err(|e| anyhow!("cache.mask({seq_len}) -> {e}"))?
                    .broadcast_as(att.shape())
                    .map_err(|e| anyhow!("mask.broadcast_as({:?}) -> {e}", att.shape()))?;

                masked_fill(&att, &mask, f32::NEG_INFINITY)
                    .map_err(|e| anyhow!("masked_fill -> {e}"))?
            };
            let att = candle_nn::ops::softmax(&att, D::Minus1)?;

            // Convert to contiguous as matmul doesn't support strided vs for now.
            att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?
        };
        

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        // log::info!("Shape of y after transpose and reshape: {:?}", y.shape());
        let y = self.o_proj.forward(&y)?;
        // log::info!("Shape of y after o_proj: {:?}", y.shape());


        Ok(y)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        candle_transformers::utils::repeat_kv(
            x,
            self.num_attention_heads / self.num_key_value_heads,
        )
    }

    /// Load an instance of this object from the VarBuilder object with the given configuration.
    pub fn load(vb: VarBuilder, cfg: &super::Config) -> Result<Self> {
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = linear(size_in, size_q, vb.pp("q_proj"))?;
        let k_proj = linear(size_in, size_kv, vb.pp("k_proj"))?;
        let v_proj = linear(size_in, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear(size_q, size_in, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
        })
    }
}
