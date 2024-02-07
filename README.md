# Random-Walk-Problem
# TD(0) Learning Experiment for Random Walk

This project implements the TD(0) learning algorithm to estimate the state values for a simple random walk environment. It explores how the algorithm's performance varies with different values of the learning rate (alpha) and the number of episodes.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [How to Use](#how-to-use)
- [File Structure](#file-structure)
- [Experiment Details](#experiment-details)
- [Results](#results)

## Introduction

TD(0) learning is a form of temporal difference learning used in reinforcement learning to estimate the value function of a Markov decision process (MDP). In this project, we apply TD(0) learning to estimate the state values for a simple random walk environment, where an agent moves left or right with equal probability.

## Installation

To run the TD(0) learning experiment for the random walk, follow these steps:

1. Clone or download the repository to your local machine.
2. Make sure you have Python installed.
3. Install the required dependencies using `pip install -r requirements.txt`.
4. Run the `RandomWalk.py` script using Python to execute the experiment.

## How to Use

1. After running the script, the experiment will generate plots showing the estimated values of different states over episodes.
2. It will also display the empirical root mean square (RMS) error, averaged over states, for different values of the learning rate (alpha) and the number of episodes.
3. Analyze the results to understand the performance of the TD(0) learning algorithm under different settings.

## File Structure

The project directory contains the following files:

- `RandomWalk.py`: Implementation of the TD(0) learning experiment for the random walk.
- `README.md`: Markdown file containing detailed information about the project.

## Experiment Details

The experiment involves estimating the state values for a random walk environment using the TD(0) learning algorithm. It explores how the choice of the learning rate (alpha) and the number of episodes impact the accuracy of the value estimates.

## Results

The experiment generates plots showing the estimated values of different states over episodes. It also displays the empirical RMS error, averaged over states, for different values of the learning rate (alpha) and the number of episodes.

