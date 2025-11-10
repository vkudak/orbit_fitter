import argparse
from io_utils import read_observations, read_tle_lines
from methods_gauss import gauss_od
from methods_laplace import laplace_od
from refine_orbit import refine_solution, compute_rms
from tle_compare import compare_with_tle

def main():
    parser = argparse.ArgumentParser(description='Orbit determination from optical measurements')
    parser.add_argument('obs_file')
    parser.add_argument('--lat', type=float, required=True)
    parser.add_argument('--lon', type=float, required=True)
    parser.add_argument('--h', type=float, default=0.0)
    parser.add_argument('--method', choices=['gauss','laplace','all'], default='all')
    parser.add_argument('--tle-line1')
    parser.add_argument('--tle-line2')
    parser.add_argument('--out-csv')
    args = parser.parse_args()

    obs = read_observations(args.obs_file)
    methods = ['gauss','laplace'] if args.method == 'all' else [args.method]
    results = {}
    for m in methods:
        print(f"Running {m}...")
        if m == 'gauss':
            state = gauss_od(obs, args.lat, args.lon, args.h)
        elif m == 'laplace':
            state = laplace_od(obs, args.lat, args.lon, args.h)
        r_opt, v_opt = refine_solution(obs, state, args.lat, args.lon, args.h)
        rms = compute_rms(obs, (r_opt, v_opt), args.lat, args.lon, args.h)
        results[m] = {'r': r_opt, 'v': v_opt, 'rms': rms}
        print(f"{m} RMS: {rms:.3f} arcsec")

    if args.tle_line1 and args.tle_line2:
        best = min(results.items(), key=lambda kv: kv[1]['rms'])
        print(f"Best method: {best[0]} with RMS {best[1]['rms']:.3f} arcsec")
        compare_with_tle(obs, (best[1]['r'], best[1]['v']), args.tle_line1, args.tle_line2, args.out_csv)

if __name__ == '__main__':
    main()
