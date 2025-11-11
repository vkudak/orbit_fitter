import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, Angle, GCRS
from astropy.coordinates import CartesianRepresentation
import astropy.units as u
from astropy.time import Time

def ra_dec_to_unitvec_icrs(ras_deg, decs_deg):
    ras = np.deg2rad(ras_deg)
    decs = np.deg2rad(decs_deg)
    x = np.cos(decs) * np.cos(ras)
    y = np.cos(decs) * np.sin(ras)
    z = np.sin(decs)
    v = np.vstack((x,y,z)).T
    v = v / np.linalg.norm(v, axis=1)[:, None]
    return v

def station_positions_gcrs(lat_deg, lon_deg, height_m, times):
    loc = EarthLocation(lat=lat_deg*u.deg, lon=lon_deg*u.deg, height=height_m*u.m)
    rs = []
    for t in times:
        g = loc.get_gcrs(t)
        rs.append(g.cartesian.xyz.to(u.km).value)
    return np.vstack(rs)


def topo_to_geo(times, ras, decs, site_lat, site_lon, site_height, assumed_range_km=500.0):
    """
    Конвертує топоцентричні RA/DEC у геоцентричні (GCRS)
    Використовується припущення про відстань до об'єкта (range), бо без неї паралакс не визначається точно.
    """
    # ініціалізуємо спостережну локацію
    loc = EarthLocation(lat=site_lat*u.deg, lon=site_lon*u.deg, height=site_height*u.m)

    # час як astropy Time
    obstime = Time(times, scale='utc')

    geo_ras = []
    geo_decs = []

    for ra, dec, t in zip(ras, decs, obstime):
        sc_topo = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs', obstime=t, location=loc)
        # задаємо умовну відстань (для геометрії)
        altaz = sc_topo.transform_to(AltAz(obstime=t, location=loc))
        sc_with_range = SkyCoord(altaz, distance=assumed_range_km*u.km)
        sc_geo = sc_with_range.transform_to('gcrs')
        geo_ras.append(sc_geo.ra.deg)
        geo_decs.append(sc_geo.dec.deg)

    return np.array(geo_ras), np.array(geo_decs)


def iterative_topo_to_geo(times, ras_deg, decs_deg, site_lat, site_lon, site_height_m,
                          init_range_km=1000.0, max_iter=6, tol_km=1e-3):
    """
    Для кожного спостереження ітеруємо range: start -> topo->geo -> new_range.
    Повертає геоцентричні RA/Dec в градусах і останні range-контексти.
    times: array-like (ISO strings or astropy Time)
    ras_deg,decs_deg: arrays degrees (топоцентричні)
    """
    obstime = Time(times, scale='utc')
    loc = EarthLocation(lat=site_lat*u.deg, lon=site_lon*u.deg, height=site_height_m*u.m)

    ras_geo = np.empty_like(ras_deg, dtype=float)
    decs_geo = np.empty_like(decs_deg, dtype=float)
    final_ranges = np.empty_like(ras_deg, dtype=float)

    for i, (t, ra_t, dec_t) in enumerate(zip(obstime, ras_deg, decs_deg)):
        # початковий range
        rho = init_range_km
        prev_rho = None

        for it in range(max_iter):
            # створюємо SkyCoord в локальному топо frame (ми знаємо RA/Dec від спостерігача)
            sc_topo = SkyCoord(ra=ra_t*u.deg, dec=dec_t*u.deg, frame='icrs', obstime=t, location=loc)
            # трансформуємо в AltAz щоб задати distance
            altaz = sc_topo.transform_to(AltAz(obstime=t, location=loc))
            sc_with_range = SkyCoord(altaz, distance=rho*u.km)
            # у GCRS (геоцентричні координати)
            sc_gcrs = sc_with_range.transform_to('gcrs')
            # позиція супутника в GCRS (в km)
            r_sat = sc_gcrs.cartesian.xyz.to(u.km).value  # shape (3,)
            # позиція спостерігача в GCRS
            r_obs = loc.get_gcrs(obstime=t).cartesian.xyz.to(u.km).value
            # новий slant-range
            new_rho = np.linalg.norm(r_sat - r_obs)
            if prev_rho is not None and abs(new_rho - prev_rho) < tol_km:
                rho = new_rho
                break
            prev_rho = rho
            rho = new_rho

        # після ітерацій беремо геоцентричні RA/Dec з r_sat
        sc_geo = SkyCoord(x=r_sat[0]*u.km, y=r_sat[1]*u.km, z=r_sat[2]*u.km,
                          frame='gcrs', representation_type='cartesian', obstime=t)
        # краще взяти gcrs->icrs для RA/Dec в інерціальній системі
        ras_geo[i] = sc_geo.icrs.ra.deg
        decs_geo[i] = sc_geo.icrs.dec.deg
        final_ranges[i] = rho

    return ras_geo, decs_geo, final_ranges


def iterative_topo_to_geo_v2(times, ras_deg, decs_deg, site_lat, site_lon, site_height_m,
                             init_range_km=1000.0, max_iter=10, tol_km=0.01, verbose=False):
    """
    Надійна ітераційна конвертація топоцентричних RA/Dec -> геоцентричні (GCRS).
    Вхід:
      times: array-like of ISO strings or astropy Time
      ras_deg, decs_deg: arrays of topocentric RA/Dec in degrees (RA in [0,360))
      site_lat, site_lon, site_height_m: station coords
      init_range_km: початкове припущення range
    Повертає:
      ras_geo_deg, decs_geo_deg, final_ranges_km, diagnostics (list)
    diagnostics[i] = dict(converged:bool, iters:int, init_range, final_range, alt_deg, az_deg)
    """
    obstime = Time(times, scale='utc')
    loc = EarthLocation(lat=site_lat*u.deg, lon=site_lon*u.deg, height=site_height_m*u.m)

    n = len(ras_deg)
    ras_geo = np.empty(n, dtype=float)
    decs_geo = np.empty(n, dtype=float)
    final_ranges = np.empty(n, dtype=float)
    diagnostics = []

    for i, (t, ra_deg, dec_deg) in enumerate(zip(obstime, ras_deg, decs_deg)):
        # Переконаємось в діапазоні RA
        ra_angle = Angle(ra_deg, unit=u.deg)
        dec_angle = Angle(dec_deg, unit=u.deg)

        # обчислимо LST та HA (в годинах -> в радіани)
        lst = t.sidereal_time('apparent', longitude=loc.lon)  # Angle
        # HA = LST - RA (wrap)
        ha = (lst - ra_angle.to(u.hourangle)).wrap_at(24*u.hourangle)
        ha_rad = ha.to(u.rad).value
        dec_rad = dec_angle.to(u.rad).value

        # обчислимо Alt, Az з HA/Dec (формули сферичної тригонометрії)
        lat_rad = np.deg2rad(site_lat)
        sin_alt = np.sin(lat_rad)*np.sin(dec_rad) + np.cos(lat_rad)*np.cos(dec_rad)*np.cos(ha_rad)
        # межі чисельності
        sin_alt = np.clip(sin_alt, -1.0, 1.0)
        alt_rad = np.arcsin(sin_alt)

        az_rad = np.arctan2(-np.sin(ha_rad),
                            np.tan(dec_rad)*np.cos(lat_rad) - np.sin(lat_rad)*np.cos(ha_rad))
        # Переведемо у градуси для діагностики
        alt_deg = np.degrees(alt_rad)
        az_deg = np.degrees(az_rad) % 360.0

        # Початковий range
        rho = float(init_range_km)
        prev_rho = None
        converged = False
        last_r_sat = None

        for it in range(1, max_iter+1):
            # Побудуємо AltAz frame з поточними alt/az та distance=rho
            altaz = AltAz(alt=alt_rad*u.rad, az=az_rad*u.rad, obstime=t, location=loc)
            sc_altaz = SkyCoord(altaz, distance=rho*u.km)
            # Трансформуємо до GCRS (геоцентричні координати)
            sc_gcrs = sc_altaz.transform_to('gcrs')
            r_sat = sc_gcrs.cartesian.xyz.to(u.km).value  # km

            # Позиція спостерігача в GCRS
            r_obs = loc.get_gcrs(obstime=t).cartesian.xyz.to(u.km).value

            # Новий slant range
            new_rho = float(np.linalg.norm(r_sat - r_obs))

            # Захист: якщо alt < -5deg — об'єкт під горизонтом або сильна неточність
            if alt_deg < -5.0:
                if verbose:
                    print(f"[frame {i}] WARNING: alt {alt_deg:.2f} deg < -5deg (object likely below horizon).")
                # все одно дозволимо ітерацію (можливо дані в топо-RA/Dec все ще корисні)
            # Перевірка зміни
            if prev_rho is not None and abs(new_rho - prev_rho) < tol_km:
                converged = True
                rho = new_rho
                last_r_sat = r_sat
                break

            prev_rho = rho
            rho = new_rho
            last_r_sat = r_sat

        # Після завершення ітерацій: запишемо результати
        if last_r_sat is None:
            # небезпечний випадок — нічого не вийшло
            ras_geo[i] = np.nan
            decs_geo[i] = np.nan
            final_ranges[i] = np.nan
        else:
            sc_geo = SkyCoord(x=last_r_sat[0]*u.km, y=last_r_sat[1]*u.km, z=last_r_sat[2]*u.km,
                              frame='gcrs', representation_type='cartesian', obstime=t)
            # Віддамо RA/Dec у ICRS (інерційні)
            ras_geo[i] = sc_geo.icrs.ra.deg
            decs_geo[i] = sc_geo.icrs.dec.deg
            final_ranges[i] = rho

        diagnostics.append({
            'converged': converged,
            'iters': it,
            'init_range_km': float(init_range_km),
            'final_range_km': float(final_ranges[i]) if not np.isnan(final_ranges[i]) else None,
            'alt_deg': float(alt_deg),
            'az_deg': float(az_deg),
            'last_r_sat': None if last_r_sat is None else list(last_r_sat),
        })

        if verbose:
            print(f"[obs {i}] RA_topo={ra_deg:.6f} dec_topo={dec_deg:.6f} alt={alt_deg:.3f} az={az_deg:.3f} "
                  f"iters={it} conv={converged} final_rho={final_ranges[i]:.3f} km")

    return ras_geo, decs_geo, final_ranges, diagnostics


def topo_to_geo_v3(times, ras, decs, site_lat, site_lon, site_h, assumed_range_km=42164.0, max_iter=30, tol=1e-6):
    """
    Перетворює топоцентричні координати RA/DEC у геоцентричні,
    використовуючи ітераційне уточнення відстані до об'єкта.

    Параметри:
    ----------
    times : array-like of str or astropy.Time
        Час спостережень (UTC)
    ras, decs : array-like
        Топоцентричні пряме піднесення (deg) і схилення (deg)
    site_n : dict
        {'lat': ..., 'lon': ..., 'height': ...}  спостережний пункт
    assumed_range_km : float
        Початкова оцінка дальності (для GEO ≈ 42164 км)
    max_iter : int
        Максимум ітерацій для уточнення
    tol : float
        Поріг збіжності по зміні відстані (у км)

    Повертає:
    ---------
    geo_ras, geo_decs : масиви (deg)
        Геоцентричні координати
    """

    # Локація спостереження
    loc = EarthLocation(
        lat=site_lat * u.deg,
        lon=site_lon * u.deg,
        height=site_h * u.m
    )

    geo_ras, geo_decs = [], []

    for t, ra_t, dec_t in zip(times, ras, decs):
        t_ast = Time(t, scale='utc')

        # 1️⃣ Початкові топоцентричні координати
        sc_topo = SkyCoord(ra=ra_t*u.deg, dec=dec_t*u.deg, frame='icrs', obstime=t_ast, location=loc)

        rho = assumed_range_km * u.km
        converged = False

        for _ in range(max_iter):
            # 2️⃣ Поточна оцінка позиції супутника у системі GCRS
            topocart = sc_topo.cartesian
            r_sat_topo = topocart * rho

            # 3️⃣ Позиція спостерігача у GCRS
            obs_gcrs = loc.get_gcrs(t_ast).cartesian

            # 4️⃣ Геоцентрична позиція супутника
            r_sat_geo = r_sat_topo + obs_gcrs

            # 5️⃣ Новий напрямок (геоцентричний)
            sc_geo = SkyCoord(CartesianRepresentation(r_sat_geo), frame=GCRS(obstime=t_ast))

            # 6️⃣ Нова відстань (оновлення ітерації)
            new_rho = np.linalg.norm(r_sat_geo.xyz - obs_gcrs.xyz)

            if abs((new_rho - rho).to_value(u.km)) < tol:
                converged = True
                break

            rho = new_rho * u.km

        geo_ras.append(sc_geo.icrs.ra.deg)
        geo_decs.append(sc_geo.icrs.dec.deg)

        print(f"[obs] RA_topo={ra_t:.6f} dec_topo={dec_t:.6f} conv={converged} final_rho={rho.value:.3f} km")

    return np.array(geo_ras), np.array(geo_decs)