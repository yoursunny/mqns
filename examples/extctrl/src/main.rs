use anyhow::Result;
use async_nats::{self, jetstream};
use bpaf::Bpaf;
use mqns_example_extctrl::{PathInstructions, Southbound};
use std::{collections::HashMap, env, time::Duration};

/// MQNS extctrl example Control Plane application
#[derive(Debug, Clone, Bpaf)]
#[bpaf(options, fallback_to_usage)]
struct Args {
    /// Prefix of NATS subjects
    #[bpaf(
        long("nats_prefix"),
        argument("PREFIX"),
        fallback(String::from("mqns.classicbridge"))
    )]
    nats_prefix: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let nats_url = env::var("NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
    let args = args().run();

    let nc = async_nats::connect(nats_url).await?;
    let js = jetstream::new(nc);

    let sb = Southbound::new(js, &args.nats_prefix);
    tx_loop(&sb).await?;

    Ok(())
}

async fn tx_loop(sb: &Southbound) -> Result<()> {
    let step = 1_000_000;
    let stop_time = 60_000_000;

    sb.stop(stop_time).await?;

    let path = PathInstructions {
        req_id: 0,
        route: vec!["S1".into(), "R1".into(), "R2".into(), "D1".into()],
        swap: vec![2, 0, 1, 2],
        swap_cutoff: vec![-1, -1, -1, -1],
        m_v: Some(vec![vec![1, 1], vec![1, 1], vec![1, 1]]),
        purif: HashMap::new(),
    };
    sb.install_path(0, 0, &path).await?;

    for t in (step..=stop_time).step_by(step as usize) {
        sb.update_gate(t).await?;
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    Ok(())
}
