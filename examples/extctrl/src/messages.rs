use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const CMD_PATH_INSERT: &str = "PATH_INSERT";
pub const CMD_PATH_DELETE: &str = "PATH_DELETE";
pub const CMD_LS: &str = "LS";

/// Path insertion command from controller to forwarders.
#[derive(Debug, Clone, Serialize)]
pub struct PathInsertMsg<'a> {
    pub cmd: &'static str, // CMD_PATH_INSERT
    pub req_id: u32,
    pub paths: &'a [PathInstructions],
}

/// Path deletion command from controller to forwarders.
#[derive(Debug, Clone, Serialize)]
pub struct PathDeleteMsg {
    pub cmd: &'static str, // CMD_PATH_DELETE
    pub req_id: u32,
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
