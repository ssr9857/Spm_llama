use std::io::{self, Write};

use crate::models::{chat::Message, Generator};
use super::Context;
use anyhow::Result;

/// 主节点和工作节点连接，通信和协调
pub struct Master<G> { 
    pub ctx: Context, 
    pub model: Box<G>,
}

/// 主节点的实现
impl<G: Generator + Send + Sync + 'static> Master<G> {
    pub async fn new(ctx: Context) -> Result<Self> {
        let model = G::load(ctx.clone()).await?;
        Ok(Self { ctx, model })
    }

    pub async fn run(mut self) -> Result<()> {
        loop {
            println!("请输入问题（输入 'q' 退出）：");
            let mut input = String::new();
            io::stdin().read_line(&mut input).expect("读取输入失败");

            let input = input.trim().to_string();

            // 检查用户是否输入 'q' 以退出循环
            if input == "q" {
                break;
            }

            // 使用 Message::user 来创建用户消息
            let message = Message::user(input);

            self.model.reset()?;
            self.model.add_message(message)?;

            // just run one generation to stdout
            self.generate(|data| {
                if data.is_empty() {
                    println!();
                } else {
                    print!("{data}");
                }
                io::stdout().flush().unwrap();
            })
            .await?;
        }

        Ok(())
    }

    /// Reset the master state for a new inference.
    /// 原本是在api/mod.rs/Responder结构体中调用的
    /// 该方法会清空分词结果、聊天历史和缓存，重置索引位置和生成的令牌数量
    pub fn reset(&mut self) -> Result<()> {
        self.model.reset()
    }

    /// Start the generation loop and call the stream function for every token.
    pub async fn generate<S>(&mut self, mut stream: S) -> Result<()>
    where
        S: FnMut(&str),
    {
        log::info!(
            "starting the inference loop (mem={})\n\n",
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        log::debug!("  ctx.args.sample_len = {}", self.ctx.args.sample_len);

        stream(&self.ctx.args.prompt);

        let mut start_gen = std::time::Instant::now();

        // for index in 0..self.ctx.args.sample_len {
        //     if index == 1 {
        //         // record start time again since the first token is the warmup
        //         start_gen = std::time::Instant::now()
        //     }

        //     let token = self.model.next_token(index).await?;
        //     if token.is_end_of_stream {
        //         break;
        //     } else {
        //         stream(&token.to_string());
        //     }
        // }
        let mut index = 0;
        loop {
            if index == 1 {
                // record start time again since the first token is the warmup
                start_gen = std::time::Instant::now()
            }

            let token = self.model.next_token(index).await?;
            if token.is_end_of_stream {
                break;
            } else {
                stream(&token.to_string());
            }
            index += 1;
        }

        // signal end of stream
        stream("");

        let dt = start_gen.elapsed();
        let generated = self.model.generated_tokens();

        log::info!(
            "{} tokens generated ({} token/s) - mem={}",
            generated,
            (generated - 1) as f64 / dt.as_secs_f64(),
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        Ok(())
    }
}