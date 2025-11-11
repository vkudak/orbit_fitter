import numpy as np
import orekit_jpype as orekit
from datetime import datetime

def setup_orekit():
    """Запуск JVM та ініціалізація Orekit"""
    orekit.initVM()
    # Додати шлях до даних Orekit, якщо потрібно
    # orekit.set_data_path('./orekit-data')
    return orekit

def orekit_od(obs, lat, lon, h, initial_state=None, make_tle=False, norad=None, cospar=None):
    """
    Функція визначення орбіти через Orekit Least Squares за RA/DEC вимірюваннями.

    obs: список/масив вимірювань [times, ras, decs, errs, mags, site_n]
    lat, lon, h: координати станції (в градусах, метри)
    initial_state: попередня оцінка орбіти (наприклад, від Gauss/Laplace)
    make_tle: чи створювати TLE після оцінки
    """
    orekit = setup_orekit()

    from org.orekit.time import AbsoluteDate, TimeScalesFactory
    from org.orekit.frames import FramesFactory
    from org.orekit.bodies import OneAxisEllipsoid, GeodeticPoint, TopocentricFrame
    from org.orekit.estimation.measurements import AngularRaDec
    from org.orekit.orbits import KeplerianOrbit
    from org.orekit.utils import Constants
    from org.orekit.estimation.leastsquares import BatchLSEstimator

    utc = TimeScalesFactory.getUTC()

    # Створюємо модель Землі
    earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                             Constants.WGS84_EARTH_FLATTENING,
                             FramesFactory.getITRF(orekit.IERSConventions.IERS_2010, True))

    # Створюємо станцію
    station_point = GeodeticPoint(np.radians(lat), np.radians(lon), h)
    station = TopocentricFrame(earth, station_point, "ObsStation")

    # Конвертуємо спостереження в AngularRaDec
    times, ras, decs, errs, *_ = obs
    measurements = []
    for t, ra_deg, dec_deg, err in zip(times, ras, decs, errs):
        # t – це datetime
        date = AbsoluteDate(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond*1e-6, utc)
        meas = AngularRaDec(station, ra_deg, dec_deg, err, err)
        measurements.append(meas)

    # Початковий стан
    if initial_state is None:
        # грубе наближення
        a = 7000e3  # наприклад, 7000 км
        e = 0.001
        i = np.radians(98)
        omega = 0.0
        raan = 0.0
        lM = 0.0
        mu = Constants.EGM96_EARTH_MU
        initial_state = KeplerianOrbit(a, e, i, omega, raan, lM,
                                       FramesFactory.getEME2000(), date, mu)

    # Least Squares оцінка
    estimator = BatchLSEstimator(initial_state, measurements)
    estimated_orbit = estimator.estimate()

    # Опціонально створюємо TLE
    if make_tle:
        # тут можна додати логіку Orekit TLE creator
        pass

    return estimated_orbit
