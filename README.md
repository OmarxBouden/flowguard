# FlowGuard
### Network-Centric Reinforcement Learning for Autonomous Cyber Defence

---

## Milestones

- [ ] **M1** — Familiarize with CybORG / CAGE 2. Train a defensive agent that performs well against the supplied red agents using a standard RL algorithm.
- [ ] **M2** — Extend CAGE 2 with network-centric observations and actions (e.g. monitor traffic, block communication between endpoints).
- [ ] **M3** — Train and compare three agents:
  - One using standard observations and standard actions
  - One using network-centric observations and standard actions
  - One using network-centric observations and network-centric actions
- [ ] **M4** *(if time allows)* — Explore abstractions over the network extensions and compare training efficiency and agent fitness.

---

## Motivation

Modern autonomous cyber defence agents typically operate at the **host level** — scanning individual machines, removing malware, and restoring systems from backup. This approach ignores a rich source of information: **the network itself**.

FlowGuard investigates whether giving a defensive RL agent visibility into network traffic flows, and the ability to act at the network level (e.g. blocking connections between endpoints), leads to meaningfully better defence outcomes.

This project builds on the [CAGE 2 Challenge](https://github.com/cage-challenge/cage-challenge-2) and the [CybORG](https://github.com/cage-challenge/CybORG) simulation environment, using [CybORG++](https://github.com/alan-turing-institute/CybORG_plus_plus) as the base.

> **Note:** Early exploratory work was started in a [separate fork](https://github.com/OmarxBouden/network-aware-cyberdefence) of the original CAGE 2 repo. Development has since moved here, building on CybORG++ for its bug fixes and cleaner codebase. MiniCAGE (also part of CybORG++) was not used, as its abstracted internals complicates extending the environment with new observations and actions.

---

## Research Questions

1. Does adding **network-centric observations** (traffic flows, connection state) improve a defensive agent's performance over standard host-level observations alone?
2. Does adding **network-centric actions** (blocking connections, isolating subnets) on top of network observations lead to further improvement?
3. What are the trade-offs in training efficiency and policy complexity across configurations?

---

## Agent Configurations

Three agents are trained and compared (M3):

| Agent | Observations | Actions |
|-------|-------------|---------|
| **Baseline** | Standard (host-level) | Standard (Analyse, Restore, Remove, Decoy) |
| **NetObs** | Network-centric + standard | Standard |
| **NetFull** | Network-centric + standard | Network-centric + standard |

Agents are trained using standard deep RL algorithms (e.g. PPO, or others as determined during M1) against the two CAGE 2 red agents (B-line and Meander), and evaluated using the official CAGE 2 evaluation protocol.

---

## Results

*To be updated later, format too is up for debate.*

| Agent | vs B-line (30) | vs B-line (50) | vs B-line (100) | vs Meander (30) | vs Meander (50) | vs Meander (100) | Total |
|-------|---------------|---------------|----------------|----------------|----------------|-----------------|-------|
| Baseline | — | — | — | — | — | — | — |
| NetObs | — | — | — | — | — | — | — |
| NetFull | — | — | — | — | — | — | — |
