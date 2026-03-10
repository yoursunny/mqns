use anyhow::Result;
use async_nats::{self, jetstream};
use bpaf::Bpaf;
use mqns_example_extctrl::{PathInstructions, Southbound};
use std::{collections::HashMap, env, time::Duration};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Bpaf)]
enum PathOpt {
    ASAP,
    L2R,
    R2L,
    Disabled,
}

impl PathOpt {
    fn to_swap(&self) -> Vec<i32> {
        match self {
            PathOpt::ASAP => vec![1, 0, 0, 1],
            PathOpt::L2R => vec![2, 0, 1, 2],
            PathOpt::R2L => vec![2, 1, 0, 2],
            _ => vec![0, 0, 0, 0],
        }
    }
}

impl std::str::FromStr for PathOpt {
    type Err = String;

    fn from_str(s: &str) -> core::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "asap" => Ok(PathOpt::ASAP),
            "l2r" => Ok(PathOpt::L2R),
            "r2l" => Ok(PathOpt::R2L),
            "disabled" => Ok(PathOpt::Disabled),
            _ => Err(format!("invalid path option: {s}")),
        }
    }
}

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

    /// Simulation accuracy in time slots per second
    #[bpaf(long("sim_accuracy"), argument("ACCURACY"), fallback(1_000_000))]
    sim_accuracy: u64,

    /// Simulation duration in seconds
    #[bpaf(long("sim_duration"), argument("DURATION"), fallback(60))]
    sim_duration: u64,

    /// S1-D1 path enablement and swap order
    #[bpaf(fallback(PathOpt::L2R))]
    path1: PathOpt,

    /// S2-D2 path enablement and swap order
    #[bpaf(fallback(PathOpt::Disabled))]
    path2: PathOpt,
}

#[tokio::main]
async fn main() -> Result<()> {
    let nats_url = env::var("NATS_URL").unwrap_or_else(|_| "nats://127.0.0.1:4222".into());
    let args = args().run();

    let nc = async_nats::connect(nats_url).await?;
    let js = jetstream::new(nc);

    let sb = Southbound::new(js, &args.nats_prefix);
    tx_loop(&args, &sb).await?;

    Ok(())
}

async fn tx_loop(args: &Args, sb: &Southbound) -> Result<()> {
    let step = 1 * args.sim_accuracy;
    let stop_time = args.sim_duration * step;

    sb.stop(stop_time).await?;

    if args.path1 != PathOpt::Disabled {
        let path = PathInstructions {
            req_id: 10,
            route: "S1-R1-R2-D1".split('-').map(String::from).collect(),
            swap: args.path1.to_swap(),
            swap_cutoff: vec![-1, -1, -1, -1],
            m_v: Some(vec![vec![1, 1], vec![1, 1], vec![1, 1]]),
            purif: HashMap::new(),
        };
        sb.install_path(0, 10, &path).await?;
    }

    if args.path2 != PathOpt::Disabled {
        let path = PathInstructions {
            req_id: 20,
            route: "S2-R1-R2-D2".split('-').map(String::from).collect(),
            swap: args.path2.to_swap(),
            swap_cutoff: vec![-1, -1, -1, -1],
            m_v: Some(vec![vec![1, 1], vec![1, 1], vec![1, 1]]),
            purif: HashMap::new(),
        };
        sb.install_path(0, 20, &path).await?;
    }

    for t in (step..=stop_time).step_by(step as usize) {
        println!("Simulation Time: {t} / {stop_time}");
        sb.update_gate(t).await?;
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    Ok(())
}
