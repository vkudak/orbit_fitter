import argparse
import sys

from constants import R_GEO, RANGE_GEO
from geometry import topo_to_geo, iterative_topo_to_geo, iterative_topo_to_geo_v2, topo_to_geo_v3
from io_utils import read_observations, read_tle_lines
from methods_gauss import gauss_od
from methods_laplace import laplace_od
from methods_orekit import orekit_od
from refine_orbit import refine_solution, compute_rms
from tle_compare import compare_with_tle

def main():
    parser = argparse.ArgumentParser(description='Orbit determination from optical measurements')
    parser.add_argument('obs_file')
    parser.add_argument('--lat', type=float, required=False, default=22.453751)
    parser.add_argument('--lon', type=float, required=False, default=48.5635505)
    parser.add_argument('--h', type=float, default=231.1325)
    parser.add_argument('--method', choices=['gauss','laplace','orekit', 'all'], default='all')
    parser.add_argument('--tle-line1')
    parser.add_argument('--tle-line2')
    parser.add_argument('--out-csv')
    args = parser.parse_args()

    norad = 15543
    cospar = '85010B'
    file_obs = read_observations(args.obs_file)
    obs_n = file_obs[norad]

    times, ras, decs, errs, mags, site_n = (
        obs_n["time"], obs_n["ra"], obs_n["dec"], obs_n["err"], obs_n["mag"], obs_n["point"]
    )
    geo_ras, geo_decs = topo_to_geo(times, ras, decs, args.lat, args.lon, args.h, assumed_range_km=RANGE_GEO)

    # geo_ras, geo_decs = topo_to_geo_v3(times, ras, decs, args.lat, args.lon, args.h, assumed_range_km=RANGE_GEO)

    obs = times, geo_ras, geo_decs, errs, mags, site_n

    # 1 28358U 04022A   23128.35604840  .00000000  00000-0  00000+0 0  9999
    # 2 28358   0.0058  35.1860 0001190  22.4337 295.5316  1.00273719 69237

    methods = ['gauss','laplace', 'orekit'] if args.method == 'all' else [args.method]
    results = {}
    state = None
    for m in methods:
        print(f"Running {m}...")
        if m == 'gauss':
            state = gauss_od(obs, args.lat, args.lon, args.h, make_tle=True, norad=norad, cospar=cospar)
        elif m == 'laplace':
            state = laplace_od(obs, args.lat, args.lon, args.h, make_tle=True, norad=norad, cospar=cospar)
        elif m == 'orekit' and state is not None:
            state = orekit_od(obs, args.lat, args.lon, args.h, make_tle=True, norad=norad, cospar=cospar,
                              initial_state=state)
        elif m == 'orekit' and state is None:
            state = orekit_od(obs, args.lat, args.lon, args.h, make_tle=True, norad=norad, cospar=cospar)


        r_opt, v_opt = refine_solution(obs, state, args.lat, args.lon, args.h)
        rms = compute_rms(obs, (r_opt, v_opt), args.lat, args.lon, args.h)

        # зберігаємо усе: r, v, rms, елементи та tle
        results[m] = {
            'r': r_opt,
            'v': v_opt,
            'rms': rms,
            'elements': state['elements'],
            'tle': state['tle']
        }

        print(f"{m} RMS: {rms:.3f} arcsec")

    # Вибір найкращого за RMS
    best = min(results.items(), key=lambda kv: kv[1]['rms'])
    best_method, best_data = best[0], best[1]

    print(f"\nBest method: {best_method} with RMS {best_data['rms']:.3f} arcsec")

    # Вивід орбітальних елементів
    print("\n=== Orbital Elements ===")
    for k, v in best_data['elements'].items():
        if k in ['i', 'raan', 'argp', 'nu']:
            print(f"{k}: {v:.3f} deg")
        elif k in ['e']:
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v:.3f} km")

    # Вивід TLE, якщо є
    if best_data['tle'] is not None:
        print("\n=== TLE ===")
        print(best_data['tle'][0])
        print(best_data['tle'][1])
    else:
        print('No TLE data')

    # Порівняння з TLE, якщо задано
    if args.tle_line1 and args.tle_line2:
        compare_with_tle(obs, (best_data['r'], best_data['v']),
                         args.tle_line1, args.tle_line2, args.out_csv)

if __name__ == '__main__':
    main()
