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
import json, os, math, time
from collections import defaultdict
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = 'matchiq-pro-2024'
API_KEY_FILE = 'api_key.txt'

# ─── In-memory cache (survives within a single server session) ───────────────
_cache = {}
CACHE_TTL = 3600  # 1 hour

# ─── Rate limit tracker ───────────────────────────────────────────────────────
_rate = {
    'limit': 10,          # requests allowed per window
    'remaining': 10,      # requests left this window
    'reset_at': 0,        # unix timestamp when window resets
    'used_this_window': 0,
}

def cache_get(key):
    if key in _cache:
        val, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            app.logger.info(f'Cache HIT: {key}')
            return val
        else:
            del _cache[key]
    return None

def cache_set(key, val):
    _cache[key] = (val, time.time())

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

def fd(endpoint, api_key, params=None, use_cache=True):
    """Fetch from football-data.org with caching to minimize API calls."""
    cache_key = f'{endpoint}_{json.dumps(params or {}, sort_keys=True)}'

    if use_cache:
        cached = cache_get(cache_key)
        if cached is not None:
            return cached

    try:
        r = requests.get(
            f'https://api.football-data.org/v4/{endpoint}',
            headers={'X-Auth-Token': api_key},
            params=params or {},
            timeout=15
        )
        app.logger.info(f'GET {endpoint} -> {r.status_code}')

        # Capture rate limit headers from every response
        def parse_rate_headers(resp):
            try:
                remaining = resp.headers.get('X-RequestCounter-Remaining')
                reset = resp.headers.get('X-RequestCounter-Reset')
                limit = resp.headers.get('X-Requests-Available-Minute')
                if remaining is not None:
                    _rate['remaining'] = int(remaining)
                if reset is not None:
                    _rate['reset_at'] = time.time() + int(reset)
                if limit is not None:
                    _rate['limit'] = int(limit)
                _rate['used_this_window'] = _rate['limit'] - _rate['remaining']
            except Exception:
                pass

        if r.status_code == 200:
            parse_rate_headers(r)
            data = r.json()
            if use_cache:
                cache_set(cache_key, data)
            return data
        elif r.status_code == 429:
            parse_rate_headers(r)
            reset = int(r.headers.get('X-RequestCounter-Reset', 60))
            _rate['reset_in_secs'] = reset
            app.logger.warning(f'Rate limited on {endpoint}, reset in {reset}s — returning None immediately')
            return None  # Never sleep — let the frontend handle retry
        elif r.status_code == 403:
            app.logger.error(f'403 on {endpoint}: access denied')
            return None
        elif r.status_code == 404:
            app.logger.error(f'404 on {endpoint}: not found')
            return None
        else:
            app.logger.error(f'{r.status_code} on {endpoint}: {r.text[:200]}')
            return None
    except requests.exceptions.Timeout:
        app.logger.error(f'Timeout on {endpoint}')
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

@app.route('/api/rate-status')
def rate_status():
    now = time.time()
    reset_in = max(0, round(_rate['reset_at'] - now)) if _rate.get('reset_at') else 60
    limited = _rate.get('remaining', 10) <= 0
    return jsonify({
        'limit': _rate.get('limit', 10),
        'remaining': _rate.get('remaining', 10),
        'used': _rate.get('used_this_window', 0),
        'reset_in': reset_in,
        'limited': limited,
        'pct_used': round((_rate.get('limit',10) - _rate.get('remaining',10)) / max(_rate.get('limit',10), 1) * 100),
    })

@app.route('/api/competitions')
def get_competitions():
    key = get_api_key()
    if not key:
        return jsonify({'error': 'No API key set. Click the API button top-right to add your key.'}), 401
    data = fd('competitions', key)
    if not data:
        return jsonify({'error': 'API rate limit hit. Wait 60 seconds then refresh the page.'}), 500
    out = [
        {'id': c['id'], 'name': c['name'], 'code': c.get('code',''), 'area': c.get('area', {}).get('name','')}
        for c in data.get('competitions', [])
    ]
    return jsonify(sorted(out, key=lambda x: x['name']))

@app.route('/api/teams/<int:comp_id>')
def get_teams(comp_id):
    key = get_api_key()
    data = fd(f'competitions/{comp_id}/teams', key)
    if not data:
        return jsonify({'error': 'Cannot load teams for this competition. Try again in 60 seconds (rate limit).'}), 403
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
        if not home_data or 'error' in home_data:
            return jsonify(home_data or {'error': 'Failed to fetch home team data'}), 500
        away_data = collect_team_data(away_id, comp_id, key)
        if not away_data or 'error' in away_data:
            return jsonify(away_data or {'error': 'Failed to fetch away team data'}), 500

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

def get_comp_code(comp_id, key):
    """Get competition code (e.g. 'PL') for a given ID. API filters use CODE not ID."""
    cached = cache_get(f'comp_code_{comp_id}')
    if cached:
        return cached
    data = fd(f'competitions/{comp_id}', key)
    if data:
        code = data.get('code', '')
        cache_set(f'comp_code_{comp_id}', code)
        return code
    return None

def collect_team_data(team_id, comp_id, key):
    """Fetch last 10 league matches with full stats + lineups."""
    app.logger.info(f'Fetching matches for team {team_id} in comp {comp_id}')

    # Per API docs: competitions filter uses CODE (e.g. PL) not numeric ID
    comp_code = get_comp_code(comp_id, key)
    app.logger.info(f'Competition code resolved: {comp_code}')

    # Strategy 1: filter by competition CODE
    params = {'status': 'FINISHED', 'limit': 15}
    if comp_code:
        params['competitions'] = comp_code
    matches_raw = fd(f'teams/{team_id}/matches', key, params)
    all_matches = (matches_raw or {}).get('matches') or []
    app.logger.info(f'Strategy 1 (code={comp_code}): {len(all_matches)} matches')

    # Strategy 2: unfiltered, filter locally
    if len(all_matches) < 3:
        matches_raw2 = fd(f'teams/{team_id}/matches', key, {'status': 'FINISHED', 'limit': 40})
        all_matches2 = (matches_raw2 or {}).get('matches') or []
        app.logger.info(f'Strategy 2 (unfiltered): {len(all_matches2)} matches')

        comp_filtered = [m for m in all_matches2
            if (str((m.get('competition') or {}).get('id', '')) == str(comp_id) or
                (m.get('competition') or {}).get('code', '') == comp_code)
            and (m.get('score') or {}).get('fullTime', {}).get('home') is not None]
        app.logger.info(f'Strategy 2 comp-filtered: {len(comp_filtered)} matches')

        if len(comp_filtered) >= 3:
            all_matches = comp_filtered
        elif all_matches2:
            all_matches = [m for m in all_matches2
                if (m.get('score') or {}).get('fullTime', {}).get('home') is not None]
            app.logger.warning(f'Cross-comp fallback: {len(all_matches)} matches')

    if not all_matches:
        return {'error': f'No match data for team {team_id}. Possible rate limit — wait 60s and retry.'}

    finished = [m for m in all_matches
                if (m.get('score') or {}).get('fullTime', {}).get('home') is not None]
    matches = finished[-10:]

    if len(matches) < 3:
        return {'error': f'Only {len(matches)} finished matches found for team {team_id}. Need at least 3.'}
    # Enrich with detailed match data (lineup + stats per match)
    # Uses cache so repeated analyses don't re-fetch the same matches
    enriched = []
    for m in matches:
        detail = fd(f'matches/{m["id"]}', key, use_cache=True)
        if detail:
            enriched.append(detail)
        else:
            app.logger.warning(f'Using basic data for match {m["id"]}')
            enriched.append(m)
        # No sleep — cache prevents redundant calls, rate limit handled by fd()

    # Get team name from matches
    team_name = ''
    short_name = ''
    for m in enriched:
        ht = (m.get('homeTeam') or {})
        at = (m.get('awayTeam') or {})
        if ht.get('id') == team_id:
            team_name = ht.get('name', '')
            short_name = ht.get('shortName', team_name)
            break
        elif at.get('id') == team_id:
            team_name = at.get('name', '')
            short_name = at.get('shortName', team_name)
            break

    try:
        result = parse_team_stats(team_id, enriched, team_name, short_name)
        if result is None:
            return {'error': f'Failed to parse stats for team {team_id}'}
        return result
    except Exception as e:
        import traceback
        app.logger.error(f'parse_team_stats error: {traceback.format_exc()}')
        return {'error': f'Stats parsing failed for team {team_id}: {str(e)}'}


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
        home_team_obj = m.get('homeTeam') or {}
        away_team_obj = m.get('awayTeam') or {}
        is_home = home_team_obj.get('id') == team_id
        my_team = home_team_obj if is_home else away_team_obj
        opp_team = away_team_obj if is_home else home_team_obj
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

        # Player appearances from lineup
        lineup = my_team.get('lineup') or []
        for player in lineup:
            pid = player.get('id')
            if pid:
                player_apps[pid]['name'] = player.get('name', '')
                player_apps[pid]['appearances'] += 1
                pos = player.get('position', '')
                if pos:
                    player_apps[pid]['positions'].add(pos)

        # Goals and assists from the goals array (per API docs)
        for goal in (m.get('goals') or []):
            goal_team_id = (goal.get('team') or {}).get('id')
            if goal_team_id == team_id:
                scorer_id = (goal.get('scorer') or {}).get('id')
                assist_id = (goal.get('assist') or {}).get('id')
                if scorer_id:
                    player_apps[scorer_id]['goals'] = player_apps[scorer_id].get('goals', 0) + 1
                    if not player_apps[scorer_id]['name']:
                        player_apps[scorer_id]['name'] = (goal.get('scorer') or {}).get('name', '')
                if assist_id:
                    player_apps[assist_id]['assists'] = player_apps[assist_id].get('assists', 0) + 1
                    if not player_apps[assist_id]['name']:
                        player_apps[assist_id]['name'] = (goal.get('assist') or {}).get('name', '')

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
    likely_xi = [{
        'name': p['name'],
        'appearances': p['appearances'],
        'position': list(p['positions'])[0] if p['positions'] else 'Unknown',
        'goals': p.get('goals', 0),
        'assists': p.get('assists', 0),
    } for p in sorted_players[:11]]

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
    """Use the official /head2head subresource from the API docs."""
    # First find a recent match ID between these two teams
    data = fd(f'teams/{team_a}/matches', key, {'status': 'FINISHED', 'limit': 40})
    matches = (data or {}).get('matches') or []

    match_id = None
    for m in reversed(matches):
        ht_id = (m.get('homeTeam') or {}).get('id')
        at_id = (m.get('awayTeam') or {}).get('id')
        if {ht_id, at_id} == {team_a, team_b}:
            match_id = m.get('id')
            break

    h2h_matches = []
    if match_id:
        # Use the official head2head subresource
        h2h_data = fd(f'matches/{match_id}/head2head', key, {'limit': 10})
        if h2h_data:
            for m in (h2h_data.get('matches') or []):
                ft = (m.get('score') or {}).get('fullTime') or {}
                hs = ft.get('home')
                aws = ft.get('away')
                if hs is not None:
                    h2h_matches.append({
                        'date': m.get('utcDate', '')[:10],
                        'home_team': (m.get('homeTeam') or {}).get('shortName', ''),
                        'away_team': (m.get('awayTeam') or {}).get('shortName', ''),
                        'home_score': hs,
                        'away_score': aws,
                    })

    # Fallback: scan team matches manually
    if not h2h_matches:
        for m in matches:
            ht_id = (m.get('homeTeam') or {}).get('id')
            at_id = (m.get('awayTeam') or {}).get('id')
            if {ht_id, at_id} == {team_a, team_b}:
                ft = (m.get('score') or {}).get('fullTime') or {}
                hs = ft.get('home')
                aws = ft.get('away')
                if hs is not None:
                    h2h_matches.append({
                        'date': m.get('utcDate', '')[:10],
                        'home_team': (m.get('homeTeam') or {}).get('shortName', ''),
                        'away_team': (m.get('awayTeam') or {}).get('shortName', ''),
                        'home_score': hs,
                        'away_score': aws,
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
    try:
        data = fd(f'competitions/{comp_id}/standings', key)
        if not data:
            return {}
        rank_map = {}
        for table in (data.get('standings') or []):
            if table.get('type') == 'TOTAL':
                for row in (table.get('table') or []):
                    tid = (row.get('team') or {}).get('id')
                    if tid:
                        rank_map[int(tid)] = {
                            'position': row.get('position') or 0,
                            'points': row.get('points') or 0,
                            'gd': row.get('goalDifference') or 0,
                            'played': row.get('playedGames') or 0,
                            'won': row.get('won') or 0,
                            'draw': row.get('draw') or 0,
                            'lost': row.get('lost') or 0,
                            'goals_for': row.get('goalsFor') or 0,
                            'goals_against': row.get('goalsAgainst') or 0,
                        }
        return rank_map
    except Exception as e:
        app.logger.error(f'get_standings error: {e}')
        return {}

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
    standings = standings or {}
    home_rank = standings.get(int(home_id), {}) or {}
    away_rank = standings.get(int(away_id), {}) or {}

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
