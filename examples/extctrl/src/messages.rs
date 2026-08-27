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
/// See `mqns.network.fw.PathInstructions` struct for details.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PathInstructions {
    pub path_id: u32,
    pub route: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bufferspace_mv: Option<Vec<u32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reactive_qubits: Option<Vec<String>>,
    pub swap: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub swap_cutoff: Option<Vec<i32>>,
    pub purif: HashMap<String, u32>,
}

impl PathInstructions {
    /// Create a builder.
    pub fn new(path_id: u32, route: Vec<String>) -> PathInstructionsBuilder {
        let n_nodes = route.len();
        PathInstructionsBuilder(PathInstructions {
            path_id,
            route,
            bufferspace_mv: None,
            reactive_qubits: None,
            swap: vec![0; n_nodes],
            swap_cutoff: None,
            purif: HashMap::new(),
        })
    }

    /// Split `"A-B-C"` to `vec!["A", "B", "C"]`.
    pub fn split_route(nodes: &str) -> Vec<String> {
        nodes.split('-').map(String::from).collect()
    }
}

pub struct PathInstructionsBuilder(PathInstructions);

impl PathInstructionsBuilder {
    /// Build the instance.
    pub fn build(self) -> PathInstructions {
        self.0
    }

    fn n_nodes(&self) -> usize {
        self.0.route.len()
    }

    pub fn bufferspace_mv(mut self, mv: Vec<u32>) -> Self {
        assert!(mv.len() == 2 * (self.n_nodes() - 1));
        self.0.bufferspace_mv = Some(mv);
        self
    }

    pub fn reactive_qubits(mut self, qubit_keys: Vec<String>) -> Self {
        assert!(qubit_keys.len() == self.n_nodes() - 1);
        self.0.reactive_qubits = Some(qubit_keys);
        self
    }

    pub fn swap(mut self, swap: Vec<u32>) -> Self {
        assert!(swap.len() == self.n_nodes());
        self.0.swap = swap;
        self
    }

    pub fn swap_cutoff(mut self, cutoff: Vec<i32>) -> Self {
        assert!(cutoff.len() == 2 * (self.n_nodes() - 2));
        self.0.swap_cutoff = Some(cutoff);
        self
    }

    pub fn purif(mut self, purif: HashMap<String, u32>) -> Self {
        self.0.purif = purif;
        self
    }
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
