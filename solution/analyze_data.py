import json, glob, numpy as np, time
from multiprocessing import Pool

# --- Configuration ---
SAMPLES = 2000  # Number of races to analyze for high precision

def load_and_preprocess():
    print("Loading and Pre-processing data...")
    races_data = []
    files = sorted(glob.glob('data/historical_races/*.json'))
    for fpath in files:
        with open(fpath) as f:
            races_data.extend(json.load(f))
        if len(races_data) >= SAMPLES: break
    
    C = {'SOFT': 0, 'MEDIUM': 1, 'HARD': 2}
    processed = []
    for race in races_data[:SAMPLES]:
        cfg = race['race_config']
        L, base, pit, temp = cfg['total_laps'], cfg['base_lap_time'], cfg['pit_lane_time'], cfg['track_temp']
        truth = race['finishing_positions']
        drivers = []
        for _, strat in race['strategies'].items():
            stops = sorted(strat['pit_stops'], key=lambda x: x['lap'])
            tire = strat['starting_tire']
            lap_start, n_pits = 1, 0
            off_w, deg_w = np.zeros(3), np.zeros(3)
            for stop in stops:
                N = stop['lap'] - lap_start + 1
                off_w[C[tire]] += N
                deg_w[C[tire]] += N*(N-1)/2
                tire = stop['to_tire']; lap_start = stop['lap']+1; n_pits += 1
            N = L - lap_start + 1
            if N > 0:
                off_w[C[tire]] += N
                deg_w[C[tire]] += N*(N-1)/2
            drivers.append({'id': strat['driver_id'], 'base_pit': base*L + n_pits*pit, 'off_w': off_w, 'deg_w': deg_w})
        
        ids = [d['id'] for d in drivers]
        processed.append({
            'off_mat': np.stack([d['off_w'] for d in drivers]),
            'deg_mat': np.stack([d['deg_w'] for d in drivers]),
            'base_vec': np.array([d['base_pit'] for d in drivers]),
            'truth_idx': np.array([ids.index(did) for did in truth]),
            'temp': float(temp)
        })
    return processed

def score_params(params_batch, processed_races):
    # params_batch shape: [Batch, 6] (S_off, H_off, dS, dM, dH, tc)
    # Returns array of scores for the batch
    B = params_batch.shape[0]
    scores = np.zeros(B, dtype=np.int32)
    
    # Pre-extract data for speed
    off_mats = np.stack([r['off_mat'] for r in processed_races]) # [R, 20, 3]
    deg_mats = np.stack([r['deg_mat'] for r in processed_races]) # [R, 20, 3]
    base_vecs = np.stack([r['base_vec'] for r in processed_races]) # [R, 20]
    truth_mat = np.stack([r['truth_idx'] for r in processed_races]) # [R, 20]
    temps = np.array([r['temp'] for r in processed_races]) # [R]

    s_off, h_off = params_batch[:, 0], params_batch[:, 1]
    ds, dm, dh, tc = params_batch[:, 2], params_batch[:, 3], params_batch[:, 4], params_batch[:, 5]

    tf = 1.0 + tc[:, None] * (temps[None, :] - 25) # [B, R]
    
    for i in range(B):
        # Calculate times for this parameter set across all races
        off_contrib = (off_mats[:, :, 0] * s_off[i] + off_mats[:, :, 2] * h_off[i])
        deg_contrib = (deg_mats[:, :, 0] * ds[i] + deg_mats[:, :, 1] * dm[i] + deg_mats[:, :, 2] * dh[i]) * tf[i, :, None]
        times = base_vecs + off_contrib + deg_contrib
        
        pred = np.argsort(times, axis=1)
        matches = np.all(pred == truth_mat, axis=1)
        scores[i] = np.sum(matches)
    return scores

if __name__ == "__main__":
    data = load_and_preprocess()
    
    # Best guess starting point
    best_p = np.array([-1.5, 1.2, 0.08, 0.03, 0.01, 0.002])
    best_score = 0
    
    # SUCCESSIVE REFINEMENT (Hill-Climbing)
    ranges = np.array([0.5, 0.5, 0.02, 0.01, 0.005, 0.001])
    
    print("Starting optimization...")
    for step in range(30):
        # Generate random candidates around the best known point
        candidates = best_p + (np.random.uniform(-1, 1, (500, 6)) * ranges)
        
        # Ensure compound logic (Soft faster than Medium, etc.)
        candidates[:, 2] = np.clip(candidates[:, 2], 0.05, 0.2) # Soft Deg
        candidates[:, 4] = np.clip(candidates[:, 4], 0.001, 0.02) # Hard Deg
        
        batch_scores = score_params(candidates, data)
        max_idx = np.argmax(batch_scores)
        
        if batch_scores[max_idx] >= best_score:
            best_score = batch_scores[max_idx]
            best_p = candidates[max_idx]
            # Shrink the search range to "zoom in"
            ranges *= 0.92 
            print(f"Step {step}: Accuracy {100*best_score/SAMPLES:.1f}% | S_off: {best_p[0]:.4f} dS: {best_p[2]:.5f}")

    print("\n=== FINAL PARAMETERS ===")
    print(f"TIRE_OFFSETS = {{'SOFT': {best_p[0]:.6f}, 'MEDIUM': 0.0, 'HARD': {best_p[1]:.6f}}}")
    print(f"DEGRADATION_RATES = {{'SOFT': {best_p[2]:.6f}, 'MEDIUM': {best_p[3]:.6f}, 'HARD': {best_p[4]:.6f}}}")
    print(f"TEMP_COEFF = {best_p[5]:.6f}")