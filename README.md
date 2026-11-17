# RL Study Desk

A single study site for learning machine learning and reinforcement learning in depth from 11 NPTEL courses (666 lectures), with a day-by-day plan, weekly assignments and practice tasks, self-tests with spaced review, final self-exams, and 20 hands-on projects ending in a modern deep-RL capstone.

**Live site:** https://prdurgam-edu.github.io/rl-study-desk/

## Courses, in study order

| Phase | Courses |
|---|---|
| 1. Foundations | Mathematical Foundations for ML (IISc) · Probability for Computer Science (IIT Kanpur) |
| 2. Stochastic processes & classical ML | Discrete-time Markov Chains and Poisson Processes (IIT Guwahati) · Introduction to Machine Learning (IIT Madras) |
| 3. Deep learning | Deep Learning (IIT Ropar) |
| 4. Core RL | Reinforcement Learning (IIT Madras) · Bandit Algorithms (IIT Bombay) |
| 5. RL theory & control | Stochastic Approximation (IISc) · Optimal Control (IIT Kharagpur) |
| 6. Multi-agent | Introduction to Multi-Agent Systems (IIT Kanpur) · Algorithmic Game Theory (IIT Kharagpur) |
| 7. Modern deep RL | Spinning Up, CS285, PPO from scratch, RLHF/DPO, capstone |

## How it works

- `index.html` is the whole app; `curriculum.json` holds every lecture (with its YouTube id), weekly assignments, tasks, readings, final-exam questions and projects.
- The plan is packed into days from your daily time budget (Setup tab) and recalculates whenever you tick something off, so unfinished work rolls forward.
- Progress and notes are saved in your browser (localStorage). Use **Setup → Copy progress backup** to move them to another device.
- `tools/` regenerates `curriculum.json`: edit `content.py`, then run `python build.py` from that folder.

Lecture videos are NPTEL's, hosted on YouTube.
