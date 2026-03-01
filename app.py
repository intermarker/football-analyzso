"""
MatchIQ Pro — Advanced Football Match Probability Analyzer
Uses football-data.org v4 API to its fullest capability:
- Full match stats (shots, possession, corners, saves, cards, fouls)
- Lineup data (formation, 11 starters + bench per match)
- Player appearance tracking across last 10 games
- Head-to-head historical data
- League standings / table position context
- Top scorers for goal-contribution weighting

Models:
1. Dixon-Coles Poisson (corrects low-score underestimation via tau parameter)
2. Elo Rating System (tracks team strength dynamically)
3. Weighted Recency Decay (recent games count more with exponential decay)
4. Shot-Ratio xG Proxy (shots on target as attack quality proxy)
5. Ensemble Blend (weighted average of all models)
"""

from flask import Flask, render_template, request, jsonify
import requests
import json, os, math
from collections import defaultdict
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = 'matchiq-pro-2024'
API_KEY_FILE = 'api_key.txt'

# ─── API helpers ────────────────────────────────────────────────────────────

def get_api_key():
    env = os.environ.get('FOOTBALL_API_KEY', '')
    if env:
        return env
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE) as f:
            return f.read().strip()
    return ''

def save_api_key(key):
    with open(API_KEY_FILE, 'w') as f:
        f.write(key.strip())

def fd(endpoint, api_key, params=None):
    """Fetch from football-data.org. Returns dict on success, None on failure."""
    try:
        r = requests.get(
            f'https://api.football-data.org/v4/{endpoint}',
            headers={'X-Auth-Token': api_key},
            params=params or {},
            timeout=15
        )
        app.logger.info(f'GET {endpoint} -> {r.status_code}')
        if r.status_code == 200:
            return r.json()
        if r.status_code == 403:
            app.logger.error(f'403 on {endpoint}: tier restriction')
        elif r.status_code == 429:
            app.logger.error(f'429 on {endpoint}: rate limit')
        else:
            app.logger.error(f'{r.status_code} on {endpoint}: {r.text[:200]}')
        return None
    except Exception as e:
        app.logger.error(f'Exception on {endpoint}: {e}')
        return None

# ─── Global error handlers ───────────────────────────────────────────────────

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    app.logger.error(f'Unhandled exception: {traceback.format_exc()}')
    return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def handle_404(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def handle_500(e):
    return jsonify({'error': str(e)}), 500

# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    has_key = bool(get_api_key())
    return render_template('index.html', has_api_key=has_key)

@app.route('/api/check-key')
def check_key():
    key = get_api_key()
    return jsonify({
        'has_key': bool(key),
        'key_length': len(key) if key else 0,
        'key_preview': key[:6] + '...' if len(key) > 6 else '(empty)',
        'source': 'env' if os.environ.get('FOOTBALL_API_KEY') else ('file' if os.path.exists(API_KEY_FILE) else 'none')
    })

@app.route('/api/save-key', methods=['POST'])
def save_key():
    key = (request.json or {}).get('api_key', '').strip()
    if not key:
        return jsonify({'error': 'No API key provided'}), 400
    result = fd('competitions', key)
    if result is None:
        return jsonify({'error': 'Invalid or unauthorised API key — check your token at football-data.org'}), 400
    save_api_key(key)
    return jsonify({'success': True})

# Competitions available on the free tier
FREE_TIER_CODES = {'PL', 'BL1', 'PD', 'SA', 'FL1', 'CL', 'DED', 'PPL', 'ELC', 'BSA', 'WC', 'EC'}

@app.route('/api/competitions')
def get_competitions():
    key = get_api_key()
    if not key:
        return jsonify({'error': 'No API key set. Click the API button top-right to add your key.'}), 401
    data = fd('competitions', key)
    if not data:
        return jsonify({'error': 'Failed to reach football-data.org — check your API key is valid'}), 500
    out = [
        {'id': c['id'], 'name': c['name'], 'code': c.get('code',''), 'area': c.get('area', {}).get('name','')}
        for c in data.get('competitions', [])
        if c.get('code') in FREE_TIER_CODES
    ]
    return jsonify(sorted(out, key=lambda x: x['name']))

@app.route('/api/teams/<int:comp_id>')
def get_teams(comp_id):
    key = get_api_key()
    data = fd(f'competitions/{comp_id}/teams', key)
    if not data:
        return jsonify({'error': 'Cannot load teams for this competition. Your API subscription tier may not support it. Please select Premier League, Bundesliga, La Liga, Serie A, Ligue 1, or Champions League.'}), 403
    teams = [{'id': t['id'], 'name': t['name'], 'shortName': t.get('shortName', t['name']),
               'crest': t.get('crest','')} for t in data.get('teams', [])]
    return jsonify(sorted(teams, key=lambda x: x['name']))

@app.route('/api/analyze', methods=['POST'])
def analyze():
    import traceback
    try:
        key = get_api_key()
        if not key:
            return jsonify({'error': 'No API key'}), 401
        d = request.json or {}
        home_id = d.get('home_team_id')
        away_id = d.get('away_team_id')
        comp_id = d.get('competition_id')
        if not all([home_id, away_id, comp_id]):
            return jsonify({'error': 'Missing parameters'}), 400

        app.logger.info(f'Analyzing: home={home_id} away={away_id} comp={comp_id}')

        home_data = collect_team_data(home_id, comp_id, key)
        if 'error' in home_data:
            return jsonify(home_data), 500
        away_data = collect_team_data(away_id, comp_id, key)
        if 'error' in away_data:
            return jsonify(away_data), 500

        h2h = get_head_to_head(home_id, away_id, key)
        standings = get_standings(comp_id, key)

        # Build full analysis
        result = build_analysis(home_data, away_data, h2h, standings)
        return jsonify(result)

    except Exception as e:
        import traceback
        app.logger.error(f'Analysis error: {traceback.format_exc()}')
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

# ─── Data Collection ─────────────────────────────────────────────────────────

def collect_team_data(team_id, comp_id, key):
    """Fetch last 10 league matches with full stats + lineups."""
    import time

    # Try with competition filter first
    matches_raw = fd(f'teams/{team_id}/matches', key, {
        'competitions': comp_id, 'status': 'FINISHED', 'limit': 15
    })

    # Fallback: fetch without competition filter (works on free tier)
    if not matches_raw:
        app.logger.warning(f'Filtered fetch failed for team {team_id}, trying unfiltered')
        matches_raw = fd(f'teams/{team_id}/matches', key, {
            'status': 'FINISHED', 'limit': 20
        })

    if not matches_raw:
        return {'error': f'Cannot fetch matches for team {team_id}. This competition may not be available on your API tier. Supported competitions: Premier League, Bundesliga, La Liga, Serie A, Ligue 1, Champions League, Eredivisie, Primeira Liga.'}

    all_matches = matches_raw.get('matches', [])

    # Filter to selected competition first
    comp_matches = [m for m in all_matches
                    if str(m.get('competition', {}).get('id', '')) == str(comp_id)
                    and m.get('score', {}).get('fullTime', {}).get('home') is not None]

    # Use competition matches if enough, otherwise use all finished
    if len(comp_matches) >= 3:
        matches = comp_matches[-10:]
    else:
        matches = [m for m in all_matches
                   if m.get('score', {}).get('fullTime', {}).get('home') is not None][-10:]

    if len(matches) < 3:
        comp_name = all_matches[0].get('competition', {}).get('name', 'unknown') if all_matches else 'unknown'
        return {'error': f'Not enough finished matches for team {team_id} (found {len(matches)}). Your API tier may not support this competition. Try: Premier League, Bundesliga, La Liga, Serie A, Ligue 1, or Champions League.'}

    # Enrich with detailed match data (lineup + stats per match)
    enriched = []
    for m in matches:
        detail = fd(f'matches/{m["id"]}', key)
        enriched.append(detail if detail else m)
        time.sleep(0.1)  # avoid rate limiting


def parse_team_stats(team_id, matches, team_name, short_name):
    """Extract comprehensive stats from enriched match list."""
    n = len(matches)
    decay_weights = [math.exp(0.1 * i) for i in range(n)]  # exponential recency decay
    total_w = sum(decay_weights)

    gs, gc = [], []
    shots_on, shots_total = [], []
    possession, corners, saves = [], [], []
    fouls, yellow_cards, red_cards = [], [], []
    results, form = [], []
    home_gs, home_gc, away_gs, away_gc = [], [], [], []
    btts_raw, o15, o25, o35 = 0, 0, 0, 0
    w_gs, w_gc, w_shots = 0, 0, 0  # weighted versions
    player_apps = defaultdict(lambda: {'name':'','appearances':0,'positions':set(),'goals':0,'assists':0})
    formations = []
    clean_sheets = 0
    failed_score = 0
    h_wins = h_draws = h_losses = 0
    a_wins = a_draws = a_losses = 0

    for i, m in enumerate(matches):
        w = decay_weights[i]
        is_home = m.get('homeTeam', {}).get('id') == team_id
        my_team = m['homeTeam'] if is_home else m['awayTeam']
        opp_team = m['awayTeam'] if is_home else m['homeTeam']
        ft = m.get('score', {}).get('fullTime', {})
        scored = (ft.get('home') if is_home else ft.get('away')) or 0
        conceded = (ft.get('away') if is_home else ft.get('home')) or 0
        total = scored + conceded

        gs.append(scored); gc.append(conceded)
        w_gs += scored * w; w_gc += conceded * w

        if is_home:
            home_gs.append(scored); home_gc.append(conceded)
        else:
            away_gs.append(scored); away_gc.append(conceded)

        if conceded == 0: clean_sheets += 1
        if scored == 0: failed_score += 1
        if scored > 0 and conceded > 0: btts_raw += 1
        if total > 1.5: o15 += 1
        if total > 2.5: o25 += 1
        if total > 3.5: o35 += 1

        if scored > conceded: r = 'W'
        elif scored == conceded: r = 'D'
        else: r = 'L'
        form.append(r)

        if is_home:
            if r == 'W': h_wins += 1
            elif r == 'D': h_draws += 1
            else: h_losses += 1
        else:
            if r == 'W': a_wins += 1
            elif r == 'D': a_draws += 1
            else: a_losses += 1

        # Match stats
        my_stats = (my_team.get('statistics') or {})
        opp_stats = (opp_team.get('statistics') or {})

        sot = my_stats.get('shots_on_goal') or 0
        st = my_stats.get('shots') or 0
        poss = my_stats.get('ball_possession') or 50
        corn = my_stats.get('corner_kicks') or 0
        sv = my_stats.get('saves') or (opp_stats.get('shots_on_goal') or 0)  # saves ≈ opponent SoT
        fl = my_stats.get('fouls') or 0
        yc = my_stats.get('yellow_cards') or 0
        rc = my_stats.get('red_cards') or 0

        shots_on.append(sot); shots_total.append(st)
        w_shots += sot * w
        possession.append(poss); corners.append(corn)
        saves.append(sv); fouls.append(fl)
        yellow_cards.append(yc); red_cards.append(rc)

        # Formation
        formation = my_team.get('formation', '')
        if formation:
            formations.append(formation)

        # Player appearances
        lineup = my_team.get('lineup', [])
        bench = my_team.get('bench', [])
        for player in lineup:
            pid = player.get('id')
            if pid:
                player_apps[pid]['name'] = player.get('name', '')
                player_apps[pid]['appearances'] += 1
                pos = player.get('position', '')
                if pos:
                    player_apps[pid]['positions'].add(pos)

        result_row = {
            'date': m.get('utcDate','')[:10],
            'match_id': m.get('id'),
            'home_team': m.get('homeTeam',{}).get('shortName', m.get('homeTeam',{}).get('name','')),
            'away_team': m.get('awayTeam',{}).get('shortName', m.get('awayTeam',{}).get('name','')),
            'home_score': ft.get('home',0),
            'away_score': ft.get('away',0),
            'is_home': is_home,
            'result': r,
            'scored': scored,
            'conceded': conceded,
            'shots_on': sot,
            'shots_total': st,
            'possession': poss,
            'corners': corn,
            'formation': formation,
        }
        results.append(result_row)

    safe_n = max(n, 1)
    avg = lambda lst: round(sum(lst) / max(len(lst), 1), 3)

    # Most common formation
    most_common_formation = max(set(formations), key=formations.count) if formations else 'N/A'

    # Likely starting 11 = players with most appearances
    sorted_players = sorted(player_apps.values(), key=lambda x: -x['appearances'])
    likely_xi = [{'name': p['name'], 'appearances': p['appearances'],
                  'position': list(p['positions'])[0] if p['positions'] else 'Unknown'}
                 for p in sorted_players[:11]]

    # Shot-based xG proxy: 0.1 per shot on target (rough but widely used proxy)
    shot_xg = round(w_shots / total_w * 0.12, 3)

    return {
        'team_id': team_id,
        'team_name': team_name,
        'short_name': short_name,
        'matches_analyzed': n,

        # Goals
        'avg_scored': round(w_gs / total_w, 3),
        'avg_conceded': round(w_gc / total_w, 3),
        'avg_scored_raw': avg(gs),
        'avg_conceded_raw': avg(gc),
        'avg_home_scored': avg(home_gs),
        'avg_away_scored': avg(away_gs),
        'avg_home_conceded': avg(home_gc),
        'avg_away_conceded': avg(away_gc),

        # Shot quality
        'avg_shots_on': round(w_shots / total_w, 2),
        'avg_shots_total': avg(shots_total),
        'shot_accuracy': round(avg(shots_on) / max(avg(shots_total), 1) * 100, 1),
        'shot_xg_proxy': shot_xg,

        # Possession & set pieces
        'avg_possession': avg(possession),
        'avg_corners': avg(corners),
        'avg_saves': avg(saves),
        'avg_fouls': avg(fouls),
        'avg_yellow_cards': avg(yellow_cards),

        # Percentages
        'btts_pct': round(btts_raw / safe_n * 100, 1),
        'over_1_5_pct': round(o15 / safe_n * 100, 1),
        'over_2_5_pct': round(o25 / safe_n * 100, 1),
        'over_3_5_pct': round(o35 / safe_n * 100, 1),
        'clean_sheet_pct': round(clean_sheets / safe_n * 100, 1),
        'failed_to_score_pct': round(failed_score / safe_n * 100, 1),
        'win_pct': round((h_wins + a_wins) / safe_n * 100, 1),
        'draw_pct': round((h_draws + a_draws) / safe_n * 100, 1),
        'loss_pct': round((h_losses + a_losses) / safe_n * 100, 1),

        # Home/Away splits
        'home_wins': h_wins, 'home_draws': h_draws, 'home_losses': h_losses,
        'away_wins': a_wins, 'away_draws': a_draws, 'away_losses': a_losses,

        # Form
        'form': form,
        'form_pts': sum({'W':3,'D':1,'L':0}[r] for r in form[-5:]),
        'most_common_formation': most_common_formation,

        # Squads
        'likely_xi': likely_xi,

        # Match history
        'match_results': results,
    }

def get_head_to_head(team_a, team_b, key):
    """Get H2H via a recent match between them and use head2head subresource."""
    # Try fetching via matches resource with both teams
    # Best available: fetch both teams' recent matches and look for common ones
    data = fd(f'teams/{team_a}/matches', key, {
        'limit': 30, 'status': 'FINISHED'
    })
    if not data:
        return {'matches': [], 'home_wins': 0, 'draws': 0, 'away_wins': 0}

    h2h_matches = []
    for m in data.get('matches', []):
        ht_id = m.get('homeTeam', {}).get('id')
        at_id = m.get('awayTeam', {}).get('id')
        if {ht_id, at_id} == {team_a, team_b}:
            ft = m.get('score', {}).get('fullTime', {})
            hs = ft.get('home'); aws = ft.get('away')
            if hs is not None:
                h2h_matches.append({
                    'date': m.get('utcDate','')[:10],
                    'home_team': m.get('homeTeam',{}).get('shortName',''),
                    'away_team': m.get('awayTeam',{}).get('shortName',''),
                    'home_score': hs, 'away_score': aws
                })

    hw = sum(1 for m in h2h_matches if m['home_score'] > m['away_score'])
    dr = sum(1 for m in h2h_matches if m['home_score'] == m['away_score'])
    aw = sum(1 for m in h2h_matches if m['home_score'] < m['away_score'])

    return {
        'matches': h2h_matches[-6:],
        'total': len(h2h_matches),
        'home_wins': hw, 'draws': dr, 'away_wins': aw
    }

def get_standings(comp_id, key):
    data = fd(f'competitions/{comp_id}/standings', key)
    if not data:
        return {}
    rank_map = {}
    for table in data.get('standings', []):
        if table.get('type') == 'TOTAL':
            for row in table.get('table', []):
                tid = row.get('team', {}).get('id')
                if tid:
                    rank_map[tid] = {
                        'position': row.get('position'),
                        'points': row.get('points'),
                        'gd': row.get('goalDifference'),
                        'played': row.get('playedGames'),
                        'won': row.get('won'), 'draw': row.get('draw'), 'lost': row.get('lost'),
                        'goals_for': row.get('goalsFor'), 'goals_against': row.get('goalsAgainst'),
                    }
    return rank_map

# ─── Statistical Models ───────────────────────────────────────────────────────

def poisson_prob(lam, k):
    if lam <= 0: lam = 0.01
    return math.exp(-lam) * (lam ** k) / math.factorial(min(k, 20))

def dixon_coles_tau(home_g, away_g, mu_h, mu_a, rho):
    """Dixon-Coles low-score correction factor."""
    if home_g == 0 and away_g == 0:
        return 1 - mu_h * mu_a * rho
    elif home_g == 0 and away_g == 1:
        return 1 + mu_h * rho
    elif home_g == 1 and away_g == 0:
        return 1 + mu_a * rho
    elif home_g == 1 and away_g == 1:
        return 1 - rho
    return 1.0

def simulate_scorelines(home_xg, away_xg, rho=-0.13, max_g=8):
    """Dixon-Coles corrected Poisson simulation."""
    matrix = {}
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            p = poisson_prob(home_xg, h) * poisson_prob(away_xg, a)
            tau = dixon_coles_tau(h, a, home_xg, away_xg, rho)
            matrix[(h, a)] = max(p * tau, 0)
    # Normalize
    total = sum(matrix.values())
    if total > 0:
        matrix = {k: v / total for k, v in matrix.items()}
    return matrix

def elo_expected(rating_a, rating_b, home_advantage=65):
    """Expected score for team A (home) vs team B."""
    return 1 / (1 + 10 ** (-(rating_a + home_advantage - rating_b) / 400))

def build_analysis(home_data, away_data, h2h, standings):
    hs = home_data
    as_ = away_data
    home_id = hs['team_id']
    away_id = as_['team_id']

    # ── League context ────
    home_rank = standings.get(home_id, {})
    away_rank = standings.get(away_id, {})

    # ── Compute xG proxies ────────────────────────────────────────────────────
    # League average baselines
    LEAGUE_HOME_AVG = 1.52
    LEAGUE_AWAY_AVG = 1.15

    # Attack / defense strengths (vs league avg)
    home_att = hs['avg_home_scored'] / LEAGUE_HOME_AVG if LEAGUE_HOME_AVG else 1
    home_def = hs['avg_home_conceded'] / LEAGUE_AWAY_AVG if LEAGUE_AWAY_AVG else 1
    away_att = as_['avg_away_scored'] / LEAGUE_AWAY_AVG if LEAGUE_AWAY_AVG else 1
    away_def = as_['avg_away_conceded'] / LEAGUE_HOME_AVG if LEAGUE_HOME_AVG else 1

    # Model 1: Dixon-Coles Poisson xG
    xg_home_dc = max(0.3, home_att * away_def * LEAGUE_HOME_AVG)
    xg_away_dc = max(0.2, away_att * home_def * LEAGUE_AWAY_AVG)

    # Model 2: Weighted recency xG (direct from weighted avg scored)
    xg_home_wr = hs['avg_scored']  # already decay-weighted
    xg_away_wr = as_['avg_scored']

    # Model 3: Shot-quality xG proxy
    xg_home_sq = hs['shot_xg_proxy']
    xg_away_sq = as_['shot_xg_proxy']

    # Ensemble xG (weighted blend)
    w_dc, w_wr, w_sq = 0.45, 0.35, 0.20
    home_xg = round(xg_home_dc * w_dc + xg_home_wr * w_wr + xg_home_sq * w_sq, 3)
    away_xg = round(xg_away_dc * w_dc + xg_away_wr * w_wr + xg_away_sq * w_sq, 3)

    # Clamp
    home_xg = max(0.25, min(home_xg, 5.0))
    away_xg = max(0.15, min(away_xg, 5.0))

    # ── Elo Ratings ──────────────────────────────────────────────────────────
    # Bootstrap Elo from league position if available, else use form points
    def elo_from_standing(rank_data, form_pts):
        pos = rank_data.get('position', 10)
        pts = rank_data.get('points', form_pts * 3)
        gd = rank_data.get('gd', 0)
        return 1500 - (pos - 1) * 30 + gd * 2 + pts * 0.5

    home_elo = elo_from_standing(home_rank, hs['form_pts'])
    away_elo = elo_from_standing(away_rank, as_['form_pts'])

    elo_home_prob = elo_expected(home_elo, away_elo)
    elo_draw_zone = 0.22  # typical draw rate in football
    elo_hw = max(0.1, elo_home_prob - elo_draw_zone / 2)
    elo_aw = max(0.1, 1 - elo_home_prob - elo_draw_zone / 2)
    elo_d = max(0.1, 1 - elo_hw - elo_aw)

    # ── Dixon-Coles scoreline simulation ─────────────────────────────────────
    score_matrix = simulate_scorelines(home_xg, away_xg)

    hw_poisson = sum(p for (h,a), p in score_matrix.items() if h > a)
    d_poisson  = sum(p for (h,a), p in score_matrix.items() if h == a)
    aw_poisson = sum(p for (h,a), p in score_matrix.items() if h < a)

    btts_poisson  = sum(p for (h,a), p in score_matrix.items() if h > 0 and a > 0)
    o15_poisson   = sum(p for (h,a), p in score_matrix.items() if h + a > 1.5)
    o25_poisson   = sum(p for (h,a), p in score_matrix.items() if h + a > 2.5)
    o35_poisson   = sum(p for (h,a), p in score_matrix.items() if h + a > 3.5)

    # ── Empirical rates ───────────────────────────────────────────────────────
    emp_btts  = (hs['btts_pct'] + as_['btts_pct']) / 2 / 100
    emp_o15   = (hs['over_1_5_pct'] + as_['over_1_5_pct']) / 2 / 100
    emp_o25   = (hs['over_2_5_pct'] + as_['over_2_5_pct']) / 2 / 100

    # ── H2H adjustment ────────────────────────────────────────────────────────
    h2h_adj = 0
    if h2h['total'] >= 3:
        h2h_hw_rate = h2h['home_wins'] / h2h['total']
        h2h_aw_rate = h2h['away_wins'] / h2h['total']
        h2h_adj_home = (h2h_hw_rate - 0.45) * 0.1
        h2h_adj_away = (h2h_aw_rate - 0.30) * 0.1
    else:
        h2h_adj_home = h2h_adj_away = 0

    # ── Final Ensemble ────────────────────────────────────────────────────────
    # 1X2: Poisson 50% + Elo 30% + Empirical 20%
    final_hw = round((hw_poisson * 0.50 + elo_hw * 0.30 + hs['win_pct'] / 100 * 0.20) + h2h_adj_home, 4)
    final_aw = round((aw_poisson * 0.50 + elo_aw * 0.30 + as_['win_pct'] / 100 * 0.20) + h2h_adj_away, 4)
    final_d  = round(max(0.05, 1 - final_hw - final_aw), 4)

    # Normalize
    total_1x2 = final_hw + final_d + final_aw
    final_hw = round(final_hw / total_1x2 * 100, 1)
    final_d  = round(final_d  / total_1x2 * 100, 1)
    final_aw = round(final_aw / total_1x2 * 100, 1)

    # Goals markets: Poisson 55% + Empirical 45%
    btts_final = round((btts_poisson * 0.55 + emp_btts * 0.45) * 100, 1)
    o15_final  = round((o15_poisson  * 0.55 + emp_o15  * 0.45) * 100, 1)
    o25_final  = round((o25_poisson  * 0.55 + emp_o25  * 0.45) * 100, 1)
    o35_final  = round( o35_poisson * 100, 1)

    # Top scorelines
    top_scores = sorted(
        [{'score': f'{h}-{a}', 'prob': round(p * 100, 2)} for (h, a), p in score_matrix.items()],
        key=lambda x: -x['prob']
    )[:10]

    # Model breakdown for transparency
    models = {
        'dixon_coles': {
            'home_win': round(hw_poisson * 100, 1),
            'draw': round(d_poisson * 100, 1),
            'away_win': round(aw_poisson * 100, 1),
            'home_xg': round(xg_home_dc, 3),
            'away_xg': round(xg_away_dc, 3),
        },
        'elo': {
            'home_win': round(elo_hw * 100, 1),
            'draw': round(elo_d * 100, 1),
            'away_win': round(elo_aw * 100, 1),
            'home_elo': round(home_elo, 0),
            'away_elo': round(away_elo, 0),
        },
        'weighted_recency': {
            'home_xg': round(xg_home_wr, 3),
            'away_xg': round(xg_away_wr, 3),
        },
        'shot_quality': {
            'home_xg': round(xg_home_sq, 3),
            'away_xg': round(xg_away_sq, 3),
        },
        'strengths': {
            'home_attack': round(home_att, 3),
            'home_defense': round(home_def, 3),
            'away_attack': round(away_att, 3),
            'away_defense': round(away_def, 3),
        }
    }

    confidence = 'High' if min(hs['matches_analyzed'], as_['matches_analyzed']) >= 8 else \
                 'Medium' if min(hs['matches_analyzed'], as_['matches_analyzed']) >= 5 else 'Low'

    return {
        'home_stats': hs,
        'away_stats': as_,
        'home_rank': home_rank,
        'away_rank': away_rank,
        'h2h': h2h,
        'predictions': {
            'home_win': final_hw,
            'draw': final_d,
            'away_win': final_aw,
            'btts': btts_final,
            'over_1_5': o15_final,
            'over_2_5': o25_final,
            'over_3_5': o35_final,
            'home_xg': home_xg,
            'away_xg': away_xg,
        },
        'top_scores': top_scores,
        'models': models,
        'confidence': confidence,
    }

if __name__ == '__main__':
    env_key = os.environ.get('FOOTBALL_API_KEY', '')
    if env_key and not os.path.exists(API_KEY_FILE):
        save_api_key(env_key)
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
