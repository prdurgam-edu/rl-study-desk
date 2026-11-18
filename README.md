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
- By default the plan fits everything into 12 weeks and tells you how much to cover per day; if you fall behind, the daily amount rises so you still finish on time. You can switch to a fixed daily time budget in Setup.
- Progress and notes are saved in your browser (localStorage). Use **Setup → Copy progress backup** to move them to another device.
- Coding exercises are guided labs that run Python in the browser (Pyodide): small steps with blanks to fill, a Check button, hints and solutions. 67 labs: a Python warm-up for SQL users (including two labs that teach lambda functions from zero), then every coding task and project for Foundations, Markov chains, Intro to ML, Deep Learning, Reinforcement Learning and Bandit Algorithms.
- Four guided Google Colab notebooks in `notebooks/` cover the parts that need PyTorch: a CNN on CIFAR-10, a character-level LSTM, ResNet fine-tuning plus a tiny GPT, and DQN on CartPole.
- `tools/` regenerates `curriculum.json`: edit `content.py` or the `labs_*.py` files, run `python test_labs.py labs_warmup` (or any `labs_*` module) to check that every solution passes its check, `python gen_nb.py all` to rebuild and smoke-test the Colab notebooks, then run `python build.py` from that folder.

Lecture videos are NPTEL's, hosted on YouTube.
