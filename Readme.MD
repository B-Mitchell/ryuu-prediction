# Football Prophet 🏆
[![Telegram](https://img.shields.io/badge/Telegram-Join%20Group-blue?logo=telegram)](https://t.me/ryuuprediction)
A Telegram bot that predicts scorelines for upcoming football matches across major leagues and the FIFA World Cup, and posts them directly to the group.

Powered by a **Poisson / Dixon-Coles statistical model** built from each team's recent form — not vibes.

## Leagues covered

- FIFA World Cup
- Premier League
- La Liga
- Bundesliga
- Serie A
- Ligue 1
- UEFA Champions League

## How it works

Uses each team's last 10 matches to compute attack and defence rates (split by home/away), blends them to produce expected goals for each side, then builds a full scoreline probability grid. The Dixon-Coles correction is applied to better handle low-scoring outcomes. The most likely scoreline is picked alongside win/draw/loss probabilities.
