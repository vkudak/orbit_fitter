import numpy as np
import math
import os
import orekit_jpype as orekit
from jpype import JClass, JArray, JDouble


def setup_orekit_clean(data_path_dir="orekit-data"):
    # 1. Ініціалізація JVM (якщо ще не ініціалізована)
    orekit.initVM()

    # 2. Використання вбудованої функції для налаштування даних
    # Ця функція робить все, що ви робили вручну, але надійно.
    try:
        orekit.setup_orekit_data(data_path_dir)
        print(f"✅ Orekit data loaded successfully from: {data_path_dir}")
    except Exception as e:
        # Якщо використовується стара версія обгортки, спробуйте альтернативний метод
        print("⚠️ Warning: setup_orekit_data failed. Trying manual setup...")
        if not os.path.exists(data_path_dir):
            raise FileNotFoundError(f"Orekit data folder not found: {data_path_dir}")

        # Ручне налаштування (для сумісності, якщо вбудована функція відсутня)
        # DataProvidersManager = JClass("org.orekit.data.DataProvidersManager")
        DirectoryCrawler = JClass("org.orekit.data.DirectoryCrawler")
        File = JClass("java.io.File")

        # Отримуємо DataContext
        DataContext = JClass("org.orekit.data.DataContext")

        # Отримуємо Singleton DataProvidersManager через DataContext
        manager = DataContext.getDefault().getDataProvidersManager()

        # manager = DataProvidersManager()
        f_dir = File(data_path_dir)

        crawler = DirectoryCrawler(f_dir)
        manager.addProvider(crawler)
        print(f"✅ Orekit data loaded manually from: {data_path_dir}")

    # 3. Перевірка доступності UTC (Тест EOP)
    try:
        TimeScalesFactory = JClass("org.orekit.time.TimeScalesFactory")
        utc = TimeScalesFactory.getUTC()
        # Якщо TimeScalesFactory.getUTC() спрацювало, це означає, що
        # Earth Orientation Parameters (EOP) були успішно завантажені.
        print("✅ Time scale UTC loaded successfully (EOP OK)")
    except Exception as e:
        print("❌ Failed to load UTC time scale (Check EOP files):", e)
        print(
            "💡 Переконайтеся, що у вас є файл 'IAU-2000-2000A.tab' та 'EOP-MPC.txt' або 'finals2000A.all' у каталозі даних.")
        raise e

    return orekit

def orekit_od(obs, lat, lon, h, initial_state=None, make_tle=False, norad=None, cospar=None):
    """
    Функція визначення орбіти через Orekit Least Squares за RA/DEC вимірюваннями.

    obs: список/масив вимірювань [times, ras, decs, errs, mags, site_n]
    lat, lon, h: координати станції (в градусах, метри)
    initial_state: попередня оцінка орбіти (наприклад, від Gauss/Laplace)
    make_tle: чи створювати TLE після оцінки
    """
    # orekit = setup_orekit()
    orekit = setup_orekit_clean()

    # Імпорти Java-класів через JClass
    AbsoluteDate = JClass("org.orekit.time.AbsoluteDate")
    TimeScalesFactory = JClass("org.orekit.time.TimeScalesFactory")
    FramesFactory = JClass("org.orekit.frames.FramesFactory")
    OneAxisEllipsoid = JClass("org.orekit.bodies.OneAxisEllipsoid")
    GeodeticPoint = JClass("org.orekit.bodies.GeodeticPoint")
    AngularRaDec = JClass("org.orekit.estimation.measurements.AngularRaDec")
    KeplerianOrbit = JClass("org.orekit.orbits.KeplerianOrbit")
    # PositionAngle = JClass("org.orekit.orbits.KeplerianOrbit.PositionAngle")
    Constants = JClass("org.orekit.utils.Constants")
    BatchLSEstimator = JClass("org.orekit.estimation.leastsquares.BatchLSEstimator")
    GroundStation =  JClass("org.orekit.estimation.measurements.GroundStation")
    ObservableSatellite =  JClass("org.orekit.estimation.measurements.ObservableSatellite")
    TopocentricFrame = JClass("org.orekit.frames.TopocentricFrame")
    IERSConventions = JClass("org.orekit.utils.IERSConventions")
    # !!! СПРОБУЙТЕ ЦЕ !!!
    PositionAngle = JClass("org.orekit.orbits.PositionAngleType")

    # Time = JClass("org.orekit.time.Time")
    System = JClass("java.lang.System")  # Для отримання системного часу
    Instant = JClass("java.time.Instant")

    LevenbergMarquardtOptimizer = JClass(
        "org.hipparchus.optim.nonlinear.vector.leastsquares.LevenbergMarquardtOptimizer")
    # Використовуємо KeplerianPropagatorBuilder, оскільки ви використовуєте KeplerianOrbit
    KeplerianPropagatorBuilder = JClass("org.orekit.propagation.conversion.KeplerianPropagatorBuilder")

    # Час та фрейми
    utc = TimeScalesFactory.getUTC()
    eme2000 = FramesFactory.getEME2000()

    # Модель Землі
    earth = OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        FramesFactory.getITRF(JClass("org.orekit.utils.IERSConventions").IERS_2010, True)
    )

    # Створюємо топоцентричну точку спостереження
    station_gp = GeodeticPoint(np.radians(lat), np.radians(lon), h)

    # Спостереження
    times, ras, decs, errs, *_ = obs
    idx = np.argsort(times)
    times = times[idx]
    ras = ras[idx]
    decs = decs[idx]
    errs = errs[idx]
    n = len(times)
    mid = n // 2


    # ----------------
    measurements = []

    # Створюємо TopocentricFrame та GroundStation один раз
    earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                             Constants.WGS84_EARTH_FLATTENING,
                             FramesFactory.getITRF(IERSConventions.IERS_2010, True))
    station_gp = GeodeticPoint(np.radians(lat), np.radians(lon), h)
    station_frame = TopocentricFrame(earth, station_gp, "ObsStation")
    station = GroundStation(station_frame)
    satellite = ObservableSatellite(0)

    for t, ra_deg, dec_deg, err in zip(times, ras, decs, errs):
        dt = t.to_datetime()
        date = AbsoluteDate(dt.year, dt.month, dt.day,
                            dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6,
                            utc)

        ra_array = JArray(JDouble, 1)([np.radians(ra_deg)])
        dec_array = JArray(JDouble, 1)([np.radians(dec_deg)])
        sigma_array = JArray(JDouble, 1)([np.radians(err), np.radians(err)])

        meas = AngularRaDec(station, station.getBaseFrame(), date,
                            ra_array, dec_array, sigma_array, satellite)
        measurements.append(meas)
    # ----------УТВ

    # Початковий стан
    if initial_state is None:
        a = 42600e3  # грубе наближення, м
        e = 0.001
        i = np.radians(98)
        omega = 0.0
        raan = 0.0
        lM = 0.0
        mu = Constants.EGM96_EARTH_MU

        # Створення AbsoluteDate з поточного Java Instant
        instant_now = Instant.now()
        initial_date = AbsoluteDate(instant_now, utc)  # Використовує Instant та TimeScale

        initial_state = KeplerianOrbit(a, e, i, omega, raan, lM, PositionAngle.MEAN,
                                       eme2000, initial_date, mu)

    # # Least Squares оцінка
    # # estimator = BatchLSEstimator(initial_state, measurements)
    # # 1. Створюємо Оптимізатор (Використовуємо стандартний LevenbergMarquardt)
    # # Зверніть увагу, що класи Hipparchus (оптимізатор) часто беруть параметри,
    # # але можна використовувати конструктор без аргументів.
    # optimizer = LevenbergMarquardtOptimizer()
    # # optimizer = LevenbergMarquardtOptimizer(1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3)
    #
    # # 2. Створюємо Будівельник Пропагатора
    # # Потрібні: initial_state, MU, PropagationType, PositionAngle
    # # Ми вже знаємо PositionAngle.MEAN.
    # PositionAngle = JClass("org.orekit.orbits.PositionAngleType")  # Імпортовано раніше
    # # mu = Constants.EGM96_EARTH_MU  # Константа з Orekit

    propagator_builder = KeplerianPropagatorBuilder(
        initial_state,  # Початкова оцінка
        PositionAngle.MEAN,
        1.0,  # Sigma, похибка моделі (1.0 - стандартне значення)
    )


    # 3. Додаємо Будівельник до масиву (оскільки конструктор очікує масив)
    # Потрібно імпортувати JArray.
    builder_array = JArray(KeplerianPropagatorBuilder)([propagator_builder])


    # 4. Least Squares оцінка
    # Тепер конструктор відповідає очікуваній сигнатурі:
    # BatchLSEstimator(LeastSquaresOptimizer, PropagatorBuilder[])
    optimizer = LevenbergMarquardtOptimizer()
    estimator = BatchLSEstimator(optimizer, builder_array)
    estimator.setMaxIterations(50)  # Наприклад, 50 ітерацій
    estimator.setMaxEvaluations(100)  # Наприклад, 100 обчислень (завжди більше, ніж ітерацій)

    # 5. Додаємо вимірювання (це новий крок!)
    # print("Number of measurements:", len(measurements))
    for meas in measurements:
        estimator.addMeasurement(meas)

    estimated_orbit = estimator.estimate()

    # Позиція та швидкість
    pv = estimated_orbit.getPVCoordinates()
    r_vec = pv.getPosition()
    v_vec = pv.getVelocity()

    r = np.array([r_vec.getX(), r_vec.getY(), r_vec.getZ()])
    v = np.array([v_vec.getX(), v_vec.getY(), v_vec.getZ()])

    # Елементи орбіти
    a = estimated_orbit.getA()
    e = estimated_orbit.getE()
    i = estimated_orbit.getI()
    raan = estimated_orbit.getRightAscensionOfAscendingNode()
    argp = estimated_orbit.getPerigeeArgument()
    nu = estimated_orbit.getTrueAnomaly()
    M = estimated_orbit.getMeanAnomaly()

    elements = {
        "a": a,
        "e": e,
        "i": np.degrees(i),
        "raan": np.degrees(raan),
        "argp": np.degrees(argp),
        "nu": np.degrees(nu),
        "M": np.degrees(M)
    }

    # Опціонально створюємо TLE через нашу функцію
    if make_tle:
        try:
            tle = make_tle_orekit(
                a, e, i, raan, argp, M, norad=norad, cospar=cospar, epoch_jd=times[mid].jd
            )
        except Exception:
            tle = None
    else:
        tle = None

    return {
        "r": r,
        "v": v,
        "elements": elements,
        "tle": tle
    }


def make_tle_orekit(a, e, i, raan, argp, M, norad, cospar, epoch_jd):
    """
    Створення TLE з орбітальних елементів через Orekit.

    Параметри:
        a, e, i, raan, argp, M : орбітальні елементи (у метрах / радіанах)
        norad : int — номер NORAD
        cospar : str — міжнародне позначення (типу "25001A")
        epoch_jd : float — юліанська дата епохи
    """
    orekit = setup_orekit_clean()

    from org.orekit.time import AbsoluteDate, TimeScalesFactory
    from org.orekit.frames import FramesFactory
    from org.orekit.orbits import KeplerianOrbit
    from org.orekit.utils import Constants
    from org.orekit.propagation.analytical.tle import TLE, TLEPropagator

    PositionAngle = JClass("org.orekit.bodies.PositionAngle")

    # 1. Базові параметри
    utc = TimeScalesFactory.getUTC()
    frame = FramesFactory.getTEME()
    date = AbsoluteDate(epoch_jd, utc)  # автоматичне перетворення з JD

    # 2. Орбіта KeplerianOrbit
    orbit = KeplerianOrbit(
        a, e, i, argp, raan, M,
        PositionAngle.MEAN, frame, date, Constants.WGS84_EARTH_MU
    )

    # 3. Розрахунок середнього руху (об/добу)
    mean_motion = orbit.getKeplerianMeanMotion() / (2 * math.pi) * 86400.0

    # 4. Формування TLE
    satnum = int(norad)
    classification = 'U'
    int_designator = cospar
    mean_motion_dot = 0.0
    mean_motion_ddot = 0.0
    bstar = 0.0
    rev_number = 0

    tle = TLE(
        satnum, classification, int_designator,
        date, mean_motion, mean_motion_dot, mean_motion_ddot,
        e, math.degrees(i), math.degrees(raan),
        math.degrees(argp), math.degrees(M),
        bstar, rev_number
    )

    # 5. Пропагація для перевірки (опціонально)
    prop = TLEPropagator.selectExtrapolator(tle)
    pv = prop.getPVCoordinates(date, frame)

    # 6. Формування структури результату
    elements = {
        "a": a,
        "e": e,
        "i": np.degrees(i),
        "raan": np.degrees(raan),
        "argp": np.degrees(argp),
        "M": np.degrees(M),
        "mean_motion": mean_motion
    }

    return {
        "r": np.array([
            pv.getPosition().getX(),
            pv.getPosition().getY(),
            pv.getPosition().getZ()
        ]),
        "v": np.array([
            pv.getVelocity().getX(),
            pv.getVelocity().getY(),
            pv.getVelocity().getZ()
        ]),
        "elements": elements,
        "tle": tle.toString()
    }