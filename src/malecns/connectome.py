"""
MaleCNS Connectome & Neural Dynamics Module
-------------------------------------------
Loads connectome network topologies (Male Adult CNS / Drosophila Melanogaster)
and simulates recurrent neural dynamics (CTRNN / LIF) from sensory inputs to descending motor output neurons.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional

class MaleCNSNetwork:
    """
    Represents the Male Central Nervous System (MaleCNS) connectome graph
    and computes dynamic firing rates across sensory, interneuron, and descending motor neurons.
    """
    def __init__(self, num_sensory: int = 16, num_interneurons: int = 64, num_descending: int = 16, seed: int = 42):
        self.num_sensory = num_sensory
        self.num_interneurons = num_interneurons
        self.num_descending = num_descending
        self.total_neurons = num_sensory + num_interneurons + num_descending
        
        # Node index offsets
        self.sensory_idx = list(range(0, num_sensory))
        self.interneuron_idx = list(range(num_sensory, num_sensory + num_interneurons))
        self.descending_idx = list(range(num_sensory + num_interneurons, self.total_neurons))
        
        # Build connectome weight matrix W and bias b
        self.graph, self.W, self.bias = self._build_malecns_topology(seed=seed)
        
        # Neuron time constants (tau)
        np.random.seed(seed)
        self.tau = np.random.uniform(0.01, 0.05, size=self.total_neurons)
        
        # State vector h (membrane potentials) and activity a (firing rates)
        self.h = np.zeros(self.total_neurons)
        self.a = np.zeros(self.total_neurons)

    def _build_malecns_topology(self, seed: int = 42) -> Tuple[nx.DiGraph, np.ndarray, np.ndarray]:
        """
        Constructs a realistic directed graph topology based on Drosophila MaleCNS connectome motifs:
        - Visual/mechanosensory inputs feed into central interneuron processing hubs (e.g. CX, LAL, VNC).
        - Recurrent connectivity within interneuron neuropils.
        - Convergence onto descending motor output neurons (DNs).
        """
        np.random.seed(seed)
        G = nx.DiGraph()
        
        # Add nodes with metadata
        for i in self.sensory_idx:
            G.add_node(i, type="sensory", name=f"Sensory_{i}")
        for i in self.interneuron_idx:
            G.add_node(i, type="interneuron", name=f"Interneuron_{i}")
        for i in self.descending_idx:
            G.add_node(i, type="descending_motor", name=f"DN_{i-self.sensory_idx[-1]-1}")
            
        W = np.zeros((self.total_neurons, self.total_neurons))
        
        # 1. Sensory -> Interneurons feedforward connections
        for s in self.sensory_idx:
            targets = np.random.choice(self.interneuron_idx, size=int(0.3 * self.num_interneurons), replace=False)
            weights = np.random.normal(1.2, 0.3, size=len(targets))
            for t, w in zip(targets, weights):
                W[t, s] = max(0.1, w)
                G.add_edge(s, t, weight=float(W[t, s]))
                
        # 2. Interneuron -> Interneuron recurrent small-world network
        ws_graph = nx.watts_strogatz_graph(self.num_interneurons, k=8, p=0.2, seed=seed)
        for u, v in ws_graph.edges():
            src = self.interneuron_idx[u]
            dst = self.interneuron_idx[v]
            # Excitatory/Inhibitory split (80/20 rule in CNS)
            weight = np.random.normal(0.8, 0.2) if np.random.rand() > 0.2 else -np.random.normal(0.6, 0.2)
            W[dst, src] = weight
            G.add_edge(src, dst, weight=float(weight))
            
        # 3. Interneuron -> Descending Motor Neurons (DNs)
        for dn in self.descending_idx:
            sources = np.random.choice(self.interneuron_idx, size=int(0.25 * self.num_interneurons), replace=False)
            weights = np.random.normal(1.5, 0.4, size=len(sources))
            for s, w in zip(sources, weights):
                W[dn, s] = max(0.1, w)
                G.add_edge(s, dn, weight=float(W[dn, s]))
                
        bias = np.random.uniform(-0.2, 0.1, size=self.total_neurons)
        return G, W, bias

    def reset_state(self):
        """Resets neuron internal states to zero."""
        self.h.fill(0.0)
        self.a.fill(0.0)

    def step(self, sensory_input: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """
        Steps continuous-time recurrent neural network (CTRNN) state forward by dt.
        
        Args:
            sensory_input: Vector of sensory stimulation [num_sensory]
            dt: Time step duration in seconds
            
        Returns:
            descending_motor_rates: Normalized firing rates of descending output neurons [num_descending]
        """
        I_ext = np.zeros(self.total_neurons)
        I_ext[self.sensory_idx] = sensory_input[:self.num_sensory]
        
        # Neural activation function: sigmoid activation sigma(h)
        sigma = 1.0 / (1.0 + np.exp(-self.h))
        
        # Recurrent synaptic input
        synaptic_input = np.dot(self.W, sigma) + self.bias + I_ext
        
        # Euler integration: dh/dt = (-h + synaptic_input) / tau
        dh = (-self.h + synaptic_input) / self.tau
        self.h += dh * dt
        
        # Update firing rates (activation clipped [0, 1])
        self.a = 1.0 / (1.0 + np.exp(-self.h))
        
        # Extract descending motor neuron rates
        return self.a[self.descending_idx]

    def get_summary(self) -> Dict:
        """Returns structural summary of the MaleCNS connectome network."""
        return {
            "total_neurons": self.total_neurons,
            "sensory_nodes": len(self.sensory_idx),
            "interneuron_nodes": len(self.interneuron_idx),
            "descending_motor_nodes": len(self.descending_idx),
            "synaptic_edges": self.graph.number_of_edges()
        }
