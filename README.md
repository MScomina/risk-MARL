# Project
Project for the course "Reinforcement Learning" - UniTS.

## Description
This project consists in implementing the board game [Risk](https://en.wikipedia.org/wiki/Risk_(game)) as a fully interactive environment through the use of PettingZoo, then trains a collection of agents through the use of **Multi-Agent Reinforcement Learning** techniques.<br>
The goal is to explore how different learning algorithms behave when multiple autonomous agents compete on the same game board, to construct a potential solution to Risk games and to understand the different approaches to the problem.

## Map Configuration
The Risk environment is *data-driven*, it does not hard-code any specific world graph.<br>
Instead, the game world is described by a **JSON map file** placed in the `environment/` directory. This is on purpose, since it allows for easier testing on differently-sized environments. The standard default Risk environment is already implemented in the `environment/classic.json`, and will be the default one if no other map is provided.

## Instructions
TBD.