//! This is the core library where all spm logic is implemented.
#[macro_use]
extern crate anyhow;

use spm::Mode;

use clap::Parser;

pub mod spm;
pub mod models;
pub mod utils;

#[derive(Clone, Parser, Default, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// GPU device index.
    #[arg(long, default_value_t = 0)]
    pub device: usize,

    /// Mode.  默认值为主节点
    #[arg(long, default_value_t, value_enum)]
    pub mode: Mode,

    /// Worker name.
    #[arg(long)]
    pub name: Option<String>,

    /// Binding address and port for workers.
    #[arg(long, default_value = "127.0.0.1:10128")]
    pub address: String,

    /// Enable OpenAI compatible chat completion API.
    #[arg(long)]
    pub api: Option<String>,

    /// Llama3 model data path.
    #[arg(long, default_value = "/home/firefly/Documents/llama3/Meta-Llama-3-8B-Instruct")]
    pub model: String,

    /// Topology file.
    #[arg(long, default_value = "/home/firefly/Documents/llama3/Spm_llama/topology.yml")]
    pub topology: String,




    
    /// The initial prompt.
    #[arg(long, default_value = "")]
    pub prompt: String,
    /// The system prompt.
    #[arg(long, default_value = "You are a helpful AI assistant.")]
    pub system_prompt: String,
    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    pub seed: u64,
    /// The length of the sample to generate (in tokens).
    #[arg(short = 'n', long, default_value_t = 2048)]
    pub sample_len: usize,
    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 1.0)]
    pub temperature: f64,
    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    pub top_p: Option<f64>,
    /// Only sample among the top K samples.
    #[arg(long)]
    pub top_k: Option<usize>,
    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    pub repeat_penalty: f32,
    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 128)]
    pub repeat_last_n: usize,
    /// Use different dtype than f16
    #[arg(long)]
    pub dtype: Option<String>,
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    pub cpu: bool,
}
