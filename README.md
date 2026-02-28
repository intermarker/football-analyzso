# MatchIQ Pro — Advanced Football Analytics

A production-grade Flask app for deep football match probability analysis using football-data.org API v4 to its full capability.

## What Data Is Used (from football-data.org)

For each of the last 10 league games per team:
- Full scoreline (home/away goals)
- Match statistics: shots, shots on target, possession, corners, saves, fouls, yellow/red cards
- Formation used in that match
- Full starting 11 lineup (player name, position, shirt number)
- Bench players
- Coach name
- Head-to-head history between both teams
- Live league standings (position, points, GD, W/D/L)

## Statistical Models (Ensemble)

### 1. Dixon-Coles Poisson (Weight: 45% of xG)
- Calculates attack/defense strength ratings relative to league averages
- Applies tau correction factor to avoid underestimating 0-0 and 1-0 scorelines
- Simulates full 0-8 × 0-8 scoreline probability matrix
- Most rigorous academic model for football prediction

### 2. Weighted Recency Decay (Weight: 35% of xG)
- Uses exponential decay (e^0.1i) to weight recent matches higher
- Last game counts ~2.7x more than game 10 games ago
- Captures current team form more accurately than simple averages

### 3. Shot-Quality xG Proxy (Weight: 20% of xG)
- Uses shots on target as a quality-of-attack proxy
- 0.12 goals per shot on target (calibrated empirical constant)
- Rewards teams with more dangerous attacks regardless of goals scored

### 4. Elo Rating System (30% of final 1X2)
- Bootstrapped from league position, points, and goal difference
- Uses standard 400-point scale with 65-point home advantage factor
- Provides long-run team strength estimate independent of recent form

### 5. Empirical Rates Blend (45% of goals markets)
- BTTS, Over 1.5, Over 2.5 rates from actual last 10 matches
- Averaged between home and away team observed rates
- Anchors predictions in observable reality

### Final Ensemble Weights
- 1X2: Poisson 50% + Elo 30% + Historical win rate 20% + H2H adjustment
- Goals markets: Poisson simulation 55% + Empirical rates 45%

## Likely Starting XI
The app tracks every player who appeared in the starting lineup across the last 10 games and ranks them by appearances. Players with the most appearances in the starting XI are displayed as the most likely starters.

## Local Setup

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

## Deployment (Recommended: Railway.app)

1. Push folder to a GitHub repo
2. Go to railway.app → New Project → Deploy from GitHub
3. Set env variable: `FOOTBALL_API_KEY=your_token`
4. Live HTTPS URL in ~2 minutes

## Updating Your API Key
Click the API badge (top right) at any time to update your token.
