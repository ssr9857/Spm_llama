use std::{
    fmt::{Debug, Display},
    path::PathBuf,
};

use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::{
    models::llama3::{Cache, Config, LlamaConfig},
    utils, Args,
};

#[cfg(feature = "master")]
mod master;

mod client;
mod proto;
mod topology;
mod worker;

pub use client::*;
pub use proto::*;
pub use topology::*;
pub use worker::*;

#[cfg(feature = "master")]
pub use master::*;

/// 表示模式类型，默认为主模式
#[derive(clap::ValueEnum, Clone, Debug, Default)]
pub enum Mode {
    #[default]
    Master,
    Worker,
}

/// Context 结构体在项目中扮演着共享状态容器的角色，它整合了运行模型推理所需的各种关键信息和资源
/// 它包含了模型的配置、数据路径、设备信息、缓存机制等，确保在推理过程中能够高效地访问和管理这些资源
#[derive(Clone)]
pub struct Context {
    pub args: Args, // 存储命令行解析得到的参数
    pub dtype: DType, // 模型推理所使用的数据类型
    pub topology: Topology, // 模型的分布式运行拓扑信息
    pub data_path: PathBuf, // 模型数据的路径 ../Meta-Llama-3-8B-Instruct/  然后从该路径下读取所有的配置文件和模型参数
    pub device: Device, // 计算设备，如 CPU 或 GPU
    pub config: Config, // 模型的配置信息，例如哪些中检层大小和隐藏层大小
    pub cache: Cache, // 用于存储中间结果的缓存对象
    pub var_builder: VarBuilder<'static>, // 用于加载模型参数的变量构建器
}

impl Context {
    /// 创建上下文通过传入的参数
    pub fn from_args(args: Args) -> Result<Self> {
        let dtype: DType = match args.dtype.as_deref() {
            Some("f16") => DType::F16,
            Some("bf16") => DType::BF16,
            Some("f32") => DType::F32,
            Some(dtype) => bail!("unsupported dtype {dtype}"),
            None => DType::F16,
        };

        let device = utils::get_inference_device(args.cpu, args.device)
            .map_err(|e| anyhow!("can't attach to device: {:?}", e))?;

        log::info!(
            "[{:?}] dtype={:?} device={:?} mem={}",
            args.mode,
            &dtype,
            &device,
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        let data_path = PathBuf::from(&args.model);

        let config_filename = data_path.join("config.json");
        let config = LlamaConfig::from_path(&config_filename)?.into_config();

        let topology = Topology::from_path(&args.topology)?;

        let model_tensors_index: PathBuf = data_path.join("model.safetensors.index.json");
        let var_builder =
            utils::load_var_builder_from_index(model_tensors_index, dtype, device.clone())?;

        let cache = Cache::new(true, dtype, &config, &device)?;

        Ok(Context {
            args,
            dtype,
            topology,
            data_path,
            device,
            config,
            cache,
            var_builder,
        })
    }
}

/// trait 是一种定义共享行为的机制，类似于其他编程语言里的接口。它能让你指定类型需要实现的一组方法，不过并不需要实现这些方法的具体内容
/// 规定了可分片对象需要实现的方法
/// Send: 表示该类型的值可以安全地跨线程发送
/// Sync: 表示该类型的值可以安全地被多个线程同时访问
/// This is the trait that a shardable object must implement（这是可分片对象必须实现的特征）
#[async_trait]
pub trait Forwarder: Debug + Send + Sync + Display {

    /// Create an instance of this object loading the specified layer(s) from a VarBuilder.
    /// 从 VarBuilder 加载指定的层来创建对象实例
    fn load(name: String, vb: VarBuilder, cfg: &Config) -> Result<Box<Self>>
    where
        Self: Sized;

    /// Applies a forward operation to the input tensor, does not require mutability.
    /// 对输入张量执行前向传播操作，不需要可变引用
    async fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor>;

    /// Applies a forward operation to the input tensor, requires mutability.
    /// 对输入张量执行前向传播操作，需要可变引用
    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor>;

    /// Applies a batch of forward operations to the input tensor.
    /// 对输入张量执行一批前向传播操作
    async fn forward_batch(
        &mut self,
        _x: &Tensor,
        _batch: Vec<(String, usize, usize)>,
        _cache: &mut Cache,
    ) -> Result<Tensor> {
        unimplemented!()
    }

    /// Return the layer name.
    /// 返回层的名称
    fn layer_name(&self) -> &str;

    /// Return the unique identity or local.
    /// 返回唯一标识或本地标识
    fn ident(&self) -> &str {
        "local"
    }
}