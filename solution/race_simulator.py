#!/usr/bin/env python3
import json
import sys

# Replace these with the EXACT values from the optimizer output
TIRE_OFFSETS = {'SOFT': -1.805006, 'MEDIUM': 0.0, 'HARD': 1.589879}
DEGRADATION_RATES = {'SOFT': 0.054265, 'MEDIUM': 0.054206, 'HARD': 0.018816}
TEMP_COEFF = 0.004668

def simulate_race(race_config, strategies):
    base_lap_time = race_config['base_lap_time']
    pit_lane_time = race_config['pit_lane_time']
    track_temp = race_config['track_temp']
    temp_factor = 1.0 + TEMP_COEFF * (track_temp - 25)

    driver_results = []
    for pos_key, strat in strategies.items():
        current_tire = strat['starting_tire']
        pit_stops = {ps['lap']: ps['to_tire'] for ps in strat['pit_stops']}
        
        total_time = 0.0
        tire_age = 0
        
        for lap in range(1, race_config['total_laps'] + 1):
            tire_age += 1 # Age increments BEFORE calculation
            
            # Relation: Time = Base + Offset + (Rate * (Age-1) * TempFactor)
            lap_time = base_lap_time + TIRE_OFFSETS[current_tire] + \
                       (DEGRADATION_RATES[current_tire] * (tire_age - 1) * temp_factor)
            
            total_time += lap_time
            
            if lap in pit_stops:
                total_time += pit_lane_time # Constant penalty
                current_tire = pit_stops[lap]
                tire_age = 0 
        
        driver_results.append({'id': strat['driver_id'], 'total_time': total_time})

    driver_results.sort(key=lambda x: x['total_time'])
    return [d['id'] for d in driver_results]

def main():
    data = sys.stdin.read()
    if data:
        test_case = json.loads(data)
        print(json.dumps({
            "race_id": test_case['race_id'],
            "finishing_positions": simulate_race(test_case['race_config'], test_case['strategies'])
        }))

if __name__ == '__main__':
    main()