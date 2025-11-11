import csv
import numpy as np
from sgp4.api import Satrec, jday
from astropy.time import Time
from astropy.coordinates import TEME, GCRS, ICRS, CartesianRepresentation
import astropy.units as u
import warnings
from datetime import datetime, timezone


warnings.filterwarnings("ignore", category=UserWarning, append=True)

def propagate_tle_to_radec(tle1, tle2, times):
    """Прогноз координат RA/DEC з TLE на задані моменти часу."""
    sat = Satrec.twoline2rv(tle1, tle2)

    ras, decs = [], []
    for t in times:
        # перетворення часу у юліанську добу
        jd, fr = jday(t.datetime.year, t.datetime.month, t.datetime.day,
                      t.datetime.hour, t.datetime.minute, t.datetime.second + t.datetime.microsecond * 1e-6)
        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            ras.append(np.nan)
            decs.append(np.nan)
            continue

        try:
            # TEME → GCRS → ICRS
            rep = CartesianRepresentation(r[0] * u.km, r[1] * u.km, r[2] * u.km)
            teme = TEME(rep, obstime=t)
            gcrs = teme.transform_to(GCRS(obstime=t))
            icrs = gcrs.transform_to(ICRS())
            ras.append(icrs.ra.deg)
            decs.append(icrs.dec.deg)
        except Exception:
            ras.append(np.nan)
            decs.append(np.nan)

    return np.array(ras), np.array(decs)


def compare_with_tle(obs, state, tle_line1, tle_line2, out_csv=None):
    """Порівнює спостереження з прогнозом TLE."""
    times, ras_obs, decs_obs, errs, mags, _ = obs

    # Перевірка формату часу
    if not isinstance(times[0], Time):
        times = Time(times)

    # Прогноз координат TLE
    ras_tle, decs_tle = propagate_tle_to_radec(tle_line1, tle_line2, times)

    # Обчислення різниць у дугових секундах
    dra = (ras_tle - ras_obs) * 3600.0 * np.cos(np.deg2rad(decs_obs))
    ddec = (decs_tle - decs_obs) * 3600.0

    valid = np.isfinite(dra) & np.isfinite(ddec)
    if np.count_nonzero(valid) == 0:
        print("⚠️  Немає спільних валідних точок для порівняння з TLE.")
        return np.nan

    rms = np.sqrt(np.nanmean(np.hstack((dra[valid], ddec[valid])) ** 2))
    print(f"TLE comparison RMS (arcsec): {rms:.3f}")

    # Запис у CSV (опційно)
    if out_csv:
        with open(out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['time_iso', 'obs_ra_deg', 'obs_dec_deg',
                        'tle_ra_deg', 'tle_dec_deg', 'dra_arcsec', 'ddec_arcsec'])
            for i in range(len(times)):
                w.writerow([
                    times[i].iso, ras_obs[i], decs_obs[i],
                    ras_tle[i], decs_tle[i], dra[i], ddec[i]
                ])

    return rms


def elements_to_tle(elements, epoch_time, sat_name='SAT'):
    """
    NOT WORKING!!!!!
    Генерує TLE для Satrec на основі орбітальних елементів.
    elements: словник з a, e, i, raan, argp, nu
    epoch_time: astropy Time об'єкт
    """
    # Переводимо у фізичні величини для TLE
    # a у km -> n (mean motion, rev/day)
    mu = 398600.4418  # km^3/s^2
    a = elements['a']
    n = np.sqrt(mu / a**3) * 86400.0 / (2.0 * np.pi)  # rev/day

    # Створюємо "порожню" Satrec і заповнюємо через twoline2rv
    sat = Satrec()
    # Використовуємо позиційні параметри для TLE
    sat.sgp4init(
        Satrec.wgs72,       # gravity model
        'i',                # type
        99999,              # satnum
        epoch_time.jd,      # epoch
        0.0,                # bstar
        0.0,                # ndot
        0.0,                # nddot
        elements['e'],      # eccentricity
        np.radians(elements['argp']),   # argpo
        np.radians(elements['i']),      # inclo
        np.radians(elements['nu']),     # mo
        n * 2*np.pi / 86400.0,          # no_kozai
        np.radians(elements['raan'])    # nodeo
    )

    # Генеруємо рядки TLE
    line1, line2 = sat.sgp4_tle()
    return line1, line2


def tle_checksum(line):
    """
    Обчислює контрольну суму TLE для одного рядка.
    Беремо всі цифри, плюс додаємо 1 за кожний знак '-'.
    Результат mod 10.
    """
    s = 0
    for c in line[:-1]:  # останній символ — це місце для checksum
        if c.isdigit():
            s += int(c)
        elif c == '-':
            s += 1
    return str(s % 10)


def jd_to_epoch_day(jd):
    """
    Перетворює JD у (epoch_year, day_of_year.fraction)
    """
    t = Time(jd, format='jd')
    yy = t.datetime.year % 100
    doy = t.datetime.timetuple().tm_yday
    frac_day = (t.datetime.hour*3600 + t.datetime.minute*60 + t.datetime.second + t.datetime.microsecond/1e6) / 86400.0
    return yy, doy + frac_day


def elements_to_tle_manual(elements, satnum=99999, cospar="98067A",
                           epoch_jd=2454975.5, element_set_number=18,
                           rev_number=6, bstar=0.0):
    """
    Проста генерація TLE рядків вручну на основі орбітальних елементів.
    elements: словник з a(km), e, i(deg), raan(deg), argp(deg), nu(deg)
    epoch: епоха у JD
    Повертає: line1, line2 з правильними checksum
    """
    a = elements['a']
    e = elements['e']
    i = elements['i']
    raan = elements['raan']
    argp = elements['argp']
    nu = elements['nu']

    # Середній рух [rev/day]
    mu = 398600.4418  # km^3/s^2
    n = np.sqrt(mu / a ** 3) * 86400.0 / (2 * np.pi)

    # JD → epoch
    yy, epoch_day = jd_to_epoch_day(epoch_jd)

    # Форматуємо e для TLE (7 цифр без крапки)
    e_tle = int(e * 1e7)

    line1 = (
            "1" +
            f"{satnum:5d}" +  # 3–7
            "U" +
            f"{cospar:<8}" +  # 10–17
            f"{yy:02d}{epoch_day:012.8f}" +  # 19–32
            f" {0.0:10.8f}" +  # First derivative, 34–43
            " 00000-0" +  # Second derivative, 45–52
            f" {int(bstar * 1e5):07d}-0" +  # B*, 54–61
            " 0" +  # Ephemeris type, 63
            f"{element_set_number:>4}"  # Element set number, 65–68
    )
    line1 = line1[:68] + tle_checksum(line1)  # 69-й символ

    # --- Рядок 2 ---
    line2 = (
            "2" +
            f"{satnum:5d}" +  # 3–7
            f"{i:8.4f}" +  # 9–16
            f"{raan:8.4f}" +  # 18–25
            f"{e_tle:07d}" +  # 27–33
            f"{argp:8.4f}" +  # 35–42
            f"{nu:8.4f}" +  # 44–51
            f"{n:11.8f}" +  # 53–63
            f"{rev_number:5d}"  # 64–68
    )
    line2 = line2[:68] + tle_checksum(line2)

    return line1, line2

# (elements, norad=norad, cospar=cospar, epoch=times[mid].jd)
def create_tle_manual(tle_elems, norad=9999, cospar='25001A', epoch=0.0):
    """
    Генерує стандартний TLE рядок (2 рядки, 69 символів) з орбітального словника.
    """
    # Розбір COSPAR
    obj_id = cospar  # приклад: "98067A"
    launch_year = int(obj_id[:2])
    launch_num = int(obj_id[2:5])
    launch_piece = obj_id[5:]
    cospar = f"{launch_year % 100:02d}{launch_num:03d}{launch_piece}"

    # Epoch у форматі YYDDD.DDDDDDDD
    # JD → epoch
    yy, epoch_day = jd_to_epoch_day(epoch)

    epoch_str = f"{yy:02d}{epoch_day:012.8f}"

    # First derivative of mean motion
    mm_dot = float(tle_elems.get("MEAN_MOTION_DOT", 0.0))
    mm_dot_str = f"{mm_dot:.8f}"

    # Second derivative (assumed 0, format e.g., 00000-0)
    mm_ddot_str = " 00000-0"

    # Середній рух [rev/day]
    mu = 398600.4418  # km^3/s^2
    a = tle_elems['a']
    n = np.sqrt(mu / a ** 3) * 86400.0 / (2 * np.pi)

    # B* drag term
    # bstar = float(tle_elems.get("BSTAR", 0.0))
    # bstar_str = f"{int(bstar*1e5):07d}-0"
    bstar_str = ' 00000-0'

    # Line 1
    line1 = (
        f"1 {int(norad):05d}U {cospar[:-1]}{cospar[-1]:3s} "
        f"{epoch_str} {mm_dot_str} {mm_ddot_str} {bstar_str} 0 "
        f"{int(6):4d}"
    )
    line1 += tle_checksum(line1)

    # Line 2
    e_str = f"{int(float(tle_elems['e'])*1e7):07d}"
    line2 = (
        f"2 {int(norad):05d} "
        f"{float(tle_elems['i']):8.4f} "
        f"{float(tle_elems['raan']):8.4f} "
        f"{e_str} "
        f"{float(tle_elems['argp']):8.4f} "
        f"{float(tle_elems['nu']):8.4f} "
        f"{float(n):11.8f} "
        f"{int(8):4d}"
    )
    line2 += tle_checksum(line2)

    return line1, line2