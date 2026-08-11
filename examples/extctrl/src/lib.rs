use anyhow::{Result, anyhow};
use async_nats::{self, HeaderMap, jetstream};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json;
use std::{
    collections::{BTreeSet, HashMap},
    sync::Arc,
};
use tokio::sync::{Mutex, mpsc};
use tokio_stream::StreamExt;

pub const CTRL_DELAY: f64 = 5e-6;

/// Convert seconds to time slots at given accuracy.
pub fn sec_to_time_slot(sec: f64, accuracy: u64) -> u64 {
    (sec * accuracy as f64).round() as u64
}

const CMD_PATH_INSERT: &str = "PATH_INSERT";
const CMD_PATH_DELETE: &str = "PATH_DELETE";
const CMD_LS: &str = "LS";

/// Path insertion command from controller to forwarders.
#[derive(Debug, Clone, Serialize)]
struct PathInsertMsg<'a> {
    cmd: &'static str, // CMD_PATH_INSERT
    req_id: u32,
    paths: &'a [PathInstructions],
}

/// Path deletion command from controller to forwarders.
#[derive(Debug, Clone, Serialize)]
struct PathDeleteMsg {
    cmd: &'static str, // CMD_PATH_DELETE
    req_id: u32,
}

/// Swapping and purification instructions for the forwarders.
/// See mqns.network.fw.PathInstructions struct for details.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PathInstructions {
    pub path_id: u32,
    pub route: Vec<String>,
    pub swap: Vec<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub swap_cutoff: Option<Vec<i32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub m_v: Option<Vec<MultiplexingVectorElem>>,
    pub purif: HashMap<String, String>,
}

impl PathInstructions {
    /// Assign m_v with specific qubits.
    ///
    /// * `qubits`: A vector of qubit reservation keys, one per quantum channel in the route.
    pub fn set_mv_qubits(&mut self, qubits: Vec<String>) {
        self.m_v = Some(
            qubits
                .into_iter()
                .map(MultiplexingVectorElem::Key)
                .collect(),
        );
    }
}

/// Multiplexing Vector element in PathInstructions.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum MultiplexingVectorElem {
    Count(i32, i32),
    Key(String),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LinkStateMsg {
    #[serde(skip)]
    pub t: u64,
    pub cmd: String, // CMD_LS
    pub ls: Vec<LinkStateEntry>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LinkStateEntry {
    pub node: String,
    pub neighbor: String,
    pub qubit: String,
}

/// Southbound interface to interact with simulated quantum nodes.
#[derive(Clone)]
pub struct Southbound {
    js: jetstream::Context,
    nats_prefix: String,
    gate_subject: String,
    gate_stream: Arc<Mutex<jetstream::consumer::pull::Stream>>,
}

impl Southbound {
    async fn create_pull_stream(
        js: &jetstream::Context,
        subject: String,
    ) -> Result<jetstream::consumer::pull::Stream> {
        let stream_name = js.stream_by_subject(&subject).await?;
        let stream = js.get_stream(stream_name).await?;
        let consumer = stream
            .create_consumer(jetstream::consumer::pull::Config {
                filter_subject: subject,
                ..Default::default()
            })
            .await?;

        let messages = consumer.messages().await?;
        Ok(messages)
    }

    fn extract_header_t(message: &jetstream::message::Message) -> u64 {
        message
            .headers
            .as_ref()
            .and_then(|h| h.get("t"))
            .and_then(|s| s.as_str().parse::<u64>().ok())
            .unwrap_or(0)
    }

    /// Construct southbound interface.
    pub async fn new(nc: async_nats::Client, nats_prefix: &str) -> Result<Self> {
        let js = jetstream::new(nc);
        let gate_stream =
            Self::create_pull_stream(&js, format!("{}.O._.gate", nats_prefix)).await?;
        Ok(Self {
            js,
            nats_prefix: nats_prefix.into(),
            gate_subject: format!("{nats_prefix}.I._.gate"),
            gate_stream: Arc::new(Mutex::new(gate_stream)),
        })
    }

    /// Send update_gate command and wait for data plane to reach the clock gate.
    ///
    /// * `t`: Clock gate in time slots.
    pub async fn update_gate(&self, t: u64) -> Result<()> {
        let mut headers = HeaderMap::new();
        headers.insert("t", t.to_string());
        self.js
            .publish_with_headers(self.gate_subject.clone(), headers, "".into())
            .await?
            .await?;

        let mut messages = self.gate_stream.lock().await;
        while let Some(Ok(message)) = messages.next().await {
            let now = Self::extract_header_t(&message);
            message.ack().await.map_err(|e| anyhow::anyhow!(e))?;
            if now >= t {
                return Ok(());
            }
        }
        Err(anyhow::anyhow!(
            "{}.O._.gate stream ended unexpectedly",
            self.nats_prefix
        ))
    }

    /// Schedule simulation stop.
    ///
    /// * `t`: Simulation stop time in time slots.
    pub async fn stop(&self, t: u64) -> Result<()> {
        let subject = format!("{}.I._.stop", self.nats_prefix);
        let mut headers = HeaderMap::new();
        headers.insert("t", t.to_string());
        self.js
            .publish_with_headers(subject, headers, "".into())
            .await?
            .await?;
        Ok(())
    }

    /// Send PATH_INSERT command.
    ///
    /// * `t`: Command transmission time in time slots.
    /// * `req_id`: Request identifier.
    /// * `paths`: Slice of routing path instructions belonging to this request.
    pub async fn path_insert(&self, t: u64, req_id: u32, paths: &[PathInstructions]) -> Result<()> {
        let msg = PathInsertMsg {
            cmd: CMD_PATH_INSERT,
            req_id,
            paths,
        };
        let payload = Bytes::from(serde_json::to_vec(&msg)?);
        self.send_instructions(t, payload, paths).await
    }

    /// Send PATH_DELETE command.
    ///
    /// * `t`: Command transmission time in time slots.
    /// * `req_id`: Request identifier.
    /// * `paths`: Slice of routing path instructions belonging to this request.
    pub async fn path_delete(&self, t: u64, req_id: u32, paths: &[PathInstructions]) -> Result<()> {
        let msg = PathDeleteMsg {
            cmd: CMD_PATH_DELETE,
            req_id,
        };
        let payload = Bytes::from(serde_json::to_vec(&msg)?);
        self.send_instructions(t, payload, paths).await
    }

    async fn send_instructions(
        &self,
        t: u64,
        payload: Bytes,
        paths: &[PathInstructions],
    ) -> Result<()> {
        // Collect unique node names in deterministic order across all paths
        let nodes: BTreeSet<&str> = paths
            .iter()
            .flat_map(|p| p.route.iter().map(|s| s.as_str()))
            .collect();

        for dest in nodes {
            let subject = format!("{}.I.{dest}.ctrl", self.nats_prefix);
            let mut headers = HeaderMap::new();
            headers.insert("t", t.to_string());
            headers.insert("fmt", "json");
            if let Err(e) = self
                .js
                .publish_with_headers(subject, headers, payload.clone())
                .await?
                .await
            {
                return Err(anyhow!("Failed to deliver instructions to {}: {}", dest, e));
            }
        }
        Ok(())
    }

    pub async fn recv_link_states(&self, ch: mpsc::Sender<LinkStateMsg>) -> Result<()> {
        let subject = format!("{}.O.ctrl.*", self.nats_prefix);
        let mut messages = Self::create_pull_stream(&self.js, subject).await?;
        while let Some(result) = messages.next().await {
            match result {
                Ok(message) => {
                    let t = Self::extract_header_t(&message);
                    message.ack().await.map_err(|e| anyhow::anyhow!(e))?;

                    if let Ok(mut msg) = serde_json::from_slice::<LinkStateMsg>(&message.payload) {
                        if msg.cmd == CMD_LS {
                            msg.t = t;
                            if let Err(_) = ch.send(msg).await {
                                break; // channel receiver closed
                            }
                        }
                    }
                }
                Err(e) => return Err(anyhow!(e)),
            }
        }

        Ok(())
    }
}
