//! This is the spm command line utility.

use spm_core::{
    spm::{Context, Master, Mode, Worker},
    Args,
};

use anyhow::Result;
use clap::Parser;

#[tokio::main]
async fn main() -> Result<()> {
    // parse command line
    let args = Args::parse();

    // setup logging
    if std::env::var_os("RUST_LOG").is_none() {
        // set `RUST_LOG=debug` to see debug logs
        std::env::set_var("RUST_LOG", "info,tokenizers=error,actix_server=warn");
    }

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_module_path(false)
        .format_target(false)
        .init();

    // setup context
    let ctx = Context::from_args(args)?;

    // run either in master or worker mode depending on command line
    let ret = match ctx.args.mode {
        Mode::Master => {
            Master::<spm_core::models::llama3::LLama>::new(ctx)
                .await?
                .run()
                .await
        }
        Mode::Worker => {
            Worker::<spm_core::models::llama3::LLama>::new(ctx)
                .await?
                .run()
                .await
        }
    };

    

    if ret.is_err() {
        // we were possibly streaming text, add a newline before reporting the error
        println!();
        return ret;
    }

    Ok(())
}
