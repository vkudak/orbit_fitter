import os
import sys
import numpy as np
from math import radians
import jpype.imports
from jpype import JClass, JArray, JDouble, getDefaultJVMPath
# from poliastro.twobody.elements import mean_motion


# ================================================================
# 1️⃣  ІНІЦІАЛІЗАЦІЯ JVM І ЗАВАНТАЖЕННЯ OREKIT
# ================================================================
def init_jvm_orekit(orekit_dir="orekit_lib", data_dir="orekit-data"):
    """
    Ініціалізує JVM з Orekit та Hipparchus JAR-файлів.
    Викликається один раз перед роботою з Orekit через JPype.

    orekit_dir : шлях до директорії, де лежать orekit*.jar та hipparchus*.jar
    data_dir   : шлях до папки з orekit-data
    """
    if jpype.isJVMStarted():
        print("ℹ️ JVM уже запущено, ініціалізація пропущена.")
        return

    # --- 💡 Зміни тут: Визначення шляху до теки скрипта ---
    # Отримуємо абсолютний шлях до теки, де знаходиться цей Python-файл
    # Якщо це головний файл, використовуємо sys.argv[0], інакше можна використати __file__
    # Для модуля/скрипта, що викликається, зазвичай безпечно використовувати:
    if getattr(sys, 'frozen', False):
        # Якщо програма запущена як виконаний файл (наприклад, pyinstaller)
        script_dir = os.path.dirname(sys.executable)
    else:
        # Для звичайного Python-скрипта
        # Використовуємо abspath і dirname від шляху поточного файлу
        script_dir = os.path.dirname(os.path.abspath(__file__))

    # Формуємо абсолютні шляхи до папок, які повинні лежати поруч зі скриптом
    abs_orekit_dir = os.path.join(script_dir, orekit_dir)
    abs_data_dir = os.path.join(script_dir, data_dir)
    # ----------------------------------------------------

    # Пошук JAR-файлів у каталозі
    if not os.path.exists(abs_orekit_dir):
        raise FileNotFoundError(f"Каталог з JAR-файлами не знайдено: {abs_orekit_dir}")

    jar_files = [os.path.join(abs_orekit_dir, f) for f in os.listdir(abs_orekit_dir) if f.endswith(".jar")]
    if not jar_files:
        raise RuntimeError(f"У каталозі {abs_orekit_dir} не знайдено жодного .jar файлу")

    # Формуємо classpath
    classpath_sep = ";" if sys.platform.startswith("win") else ":"
    classpath = classpath_sep.join(jar_files)

    jvm_path = getDefaultJVMPath()
    print(f"🟢 Використовується JVM: {jvm_path}")
    print(f"🟢 Завантаження JAR-файлів з: {abs_orekit_dir}")

    jpype.startJVM(
        jvm_path,
        "-ea",
        "--enable-native-access=ALL-UNNAMED",
        f"-Djava.class.path={classpath}"
    )

    # Завантаження orekit-data
    from org.orekit.data import DataContext, DirectoryCrawler
    from java.io import File

    manager = DataContext.getDefault().getDataProvidersManager()
    manager.addProvider(DirectoryCrawler(File(abs_data_dir)))
    # print(manager)
    print(f"✅ Orekit data loaded successfully from: {abs_data_dir}")




def datetime_to_absolutedate(dt_utc):
    """Перетворює Python datetime (UTC) у Orekit AbsoluteDate"""
    AbsoluteDate = JClass("org.orekit.time.AbsoluteDate")
    TimeScalesFactory = JClass("org.orekit.time.TimeScalesFactory")

    utc = TimeScalesFactory.getUTC()
    # Якщо це astropy.Time — конвертуємо
    if hasattr(dt_utc, "to_datetime"):
        dt_utc = dt_utc.to_datetime()

    return AbsoluteDate(
        dt_utc.year,
        dt_utc.month,
        dt_utc.day,
        dt_utc.hour,
        dt_utc.minute,
        dt_utc.second + dt_utc.microsecond / 1e6,
        utc
    )


def orekit_od(obs, lat, lon, h, initial_state=None, make_tle=False, norad=None, cospar=None):
    """
    Визначення орбіти за вимірюваннями RA/DEC через Orekit (JPype, без імпорту orekit-модуля).
    """

    # --- Імпорт Java-класів з Orekit ---
    TimeScalesFactory = JClass("org.orekit.time.TimeScalesFactory")
    FramesFactory = JClass("org.orekit.frames.FramesFactory")
    OneAxisEllipsoid = JClass("org.orekit.bodies.OneAxisEllipsoid")
    GeodeticPoint = JClass("org.orekit.bodies.GeodeticPoint")
    AngularRaDec = JClass("org.orekit.estimation.measurements.AngularRaDec")
    ObservableSatellite = JClass("org.orekit.estimation.measurements.ObservableSatellite")
    GroundStation = JClass("org.orekit.estimation.measurements.GroundStation")
    TopocentricFrame = JClass("org.orekit.frames.TopocentricFrame")
    BatchLSEstimator = JClass("org.orekit.estimation.leastsquares.BatchLSEstimator")
    KeplerianOrbit = JClass("org.orekit.orbits.KeplerianOrbit")
    KeplerianPropagatorBuilder = JClass("org.orekit.propagation.conversion.KeplerianPropagatorBuilder")
    PositionAngleType = JClass("org.orekit.orbits.PositionAngleType")
    CartesianOrbit = JClass("org.orekit.orbits.CartesianOrbit")
    PVCoordinates = JClass("org.orekit.utils.PVCoordinates")
    Constants = JClass("org.orekit.utils.Constants")
    IERSConventions = JClass("org.orekit.utils.IERSConventions")
    Vector3D = JClass("org.hipparchus.geometry.euclidean.threed.Vector3D")
    NumericalPropagatorBuilder = JClass("org.orekit.propagation.conversion.NumericalPropagatorBuilder")
    DormandPrince853IntegratorBuilder = JClass("org.orekit.propagation.conversion.DormandPrince853IntegratorBuilder")
    LevenbergMarquardtOptimizer = JClass(
        "org.hipparchus.optim.nonlinear.vector.leastsquares.LevenbergMarquardtOptimizer")
    NewtonianAttraction = JClass("org.orekit.forces.gravity.NewtonianAttraction")
    OrbitType = JClass("org.orekit.orbits.OrbitType")
    EquinoctialOrbit = JClass("org.orekit.orbits.EquinoctialOrbit")
    AbsoluteDate = JClass('org.orekit.time.AbsoluteDate')

    CelestialBodyFactory = JClass("org.orekit.bodies.CelestialBodyFactory")
    ThirdBodyAttraction = JClass("org.orekit.forces.gravity.ThirdBodyAttraction")

    # Тиск Сонячного Випромінювання (SRP)
    SolarRadiationPressure = JClass("org.orekit.forces.radiation.SolarRadiationPressure")
    IsotropicRadiationSingleCoefficient = JClass("org.orekit.forces.radiation.IsotropicRadiationSingleCoefficient")
    OneAxisEllipsoid = JClass("org.orekit.bodies.OneAxisEllipsoid")

    UTC = TimeScalesFactory.getUTC()
    IERSConventions = JClass("org.orekit.utils.IERSConventions")
    ITRF = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    TEME = FramesFactory.getTEME()
    EME2000 = FramesFactory.getEME2000()
    MU = Constants.WGS84_EARTH_MU
    EARTH_RADIUS = Constants.WGS84_EARTH_EQUATORIAL_RADIUS

    times, ras, decs, errs, *_ = obs
    station_lla = (radians(lat), radians(lon), h)

    # Земля і станція
    earth = OneAxisEllipsoid(EARTH_RADIUS, Constants.WGS84_EARTH_FLATTENING, ITRF)
    geo_point = GeodeticPoint(*station_lla)
    station_frame = TopocentricFrame(earth, geo_point, "ObsStation")
    station = GroundStation(station_frame)
    satellite = ObservableSatellite(0)

    # Якщо початковий стан не заданий — створимо грубе коло
    date = datetime_to_absolutedate(times[0])
    if initial_state is None:
        a = EARTH_RADIUS + 35786e3
        v = np.sqrt(MU / a)
        pv = PVCoordinates(Vector3D(float(a), 0.0, 0.0), Vector3D(0.0, float(v), 0.0))
        # initial_state = CartesianOrbit(pv, TEME, date, MU)

        # cartesian_orbit = CartesianOrbit(pv, TEME, date, MU)
        # initial_state = EquinoctialOrbit(cartesian_orbit)
        initial_state = EquinoctialOrbit(pv, TEME, date, MU)
    else:
        # r2 = initial_state["r"]  # Вектор позиції (m)
        # v2 = initial_state["v"]  # Вектор швидкості (m/s)
        #
        # # Перетворення списків/масивів у Vector3D (припускаючи, що r2 і v2 - це масиви/списки з 3 елементів)
        # position = Vector3D(float(r2[0]), float(r2[1]), float(r2[2]))
        # velocity = Vector3D(float(v2[0]), float(v2[1]), float(v2[2]))
        #
        # pv = PVCoordinates(position, velocity)
        # date = datetime_to_absolutedate(times[0])
        # # initial_state = CartesianOrbit(pv, TEME, date, MU)
        # # cartesian_orbit = CartesianOrbit(pv, TEME, date, MU)
        # # initial_state = EquinoctialOrbit(cartesian_orbit)
        # initial_state = EquinoctialOrbit(pv, TEME, date, MU)

        # 1. Створення KeplerianOrbit на основі результатів Лапласа, але з e = 0.0
        # Кутова позиція - True Anomaly (nu)
        elements = initial_state['elements']
        semi_major_axis = elements['a'] * 1000 # У метрах
        eccentricity = elements['e'] #0.0001  # ❗️ Створюємо кругову орбіту
        inclination = np.radians(elements['i'])
        raan = np.radians(elements['raan'])
        arg_of_pericenter = np.radians(elements['argp'])
        true_anomaly = np.radians(elements['nu'])

        initial_keplerian = KeplerianOrbit(
            semi_major_axis,
            eccentricity,
            inclination,
            raan,
            arg_of_pericenter,
            true_anomaly,
            PositionAngleType.TRUE,  # Використовуємо True Anomaly (nu)
            TEME,
            date,
            MU
        )

        # 2. Конвертація KeplerianOrbit у стійкий EquinoctialOrbit
        # Використовуємо конструктор з одним аргументом, як ми виправляли раніше.
        initial_state = EquinoctialOrbit(initial_keplerian)
        print('new e=',initial_state.getE())


    # Імпорт потрібних класів
    HolmesFeatherstoneAttractionModel = JClass("org.orekit.forces.gravity.HolmesFeatherstoneAttractionModel")
    GravityFieldFactory = JClass(
        "org.orekit.forces.gravity.potential.GravityFieldFactory")  # Цей клас потрібен для отримання потенціалу

    # Налаштування кроків у секундах (змініть ці значення)
    min_step = 0.00000001  # Зменште мінімальний крок до меншого значення
    max_step = 1000.0
    init_step = 60.0

    # 2. Створіть толерантності
    # abs_tol = JArray(JDouble)([1.0e-15, 1.0e-15, 1.0e-15])  # Абсолютна толерантність (наприклад, 1 мм)
    # rel_tol = JArray(JDouble)([1.0e-6, 1.0e-6, 1.0e-6])  # Відносна толерантність
    # tolerance_provider = AbsoluteToleranceProvider(abs_tol, rel_tol)

    # integrator_builder = DormandPrince853IntegratorBuilder(1.0, 300.0, 1.0e-3)

    # 3. Використовуйте Builder та встановіть кроки
    integrator_builder = DormandPrince853IntegratorBuilder(
    float(min_step),
        float(max_step),
        float(init_step)
    )

    # OrbitType = JClass("org.orekit.orbits.OrbitType")

    # ❗️ Використовуйте NumericalPropagatorBuilder:
    propagator_builder = NumericalPropagatorBuilder(
        initial_state,
        integrator_builder,
        # OrbitType.EQUINOCTIAL,
        PositionAngleType.TRUE,
        1.0e-5 #0.1  # Sigma
    )

    try:
        print('Trying to add gravitation field...', end='')
        # Завантаження стандартного гравітаційного поля (наприклад, WGS84 EGM)
        # З порядком і ступенем (degree and order) 2 - це J2.
        # Використовуємо 5x5, щоб мати трохи більше точності.

        j2_provider = GravityFieldFactory.getConstantNormalizedProvider(
            8,  # Degree
            8,  # Order
            AbsoluteDate.J2000_EPOCH
        )

        force_model = HolmesFeatherstoneAttractionModel(ITRF, j2_provider)
        propagator_builder.addForceModel(force_model)
        print(" ✅ OK")
    except Exception as e:
        print(
            f"\n⚠️ Помилка при додаванні моделі J2 (гравітаційний потенціал). Спробуйте лише NewtonianAttraction. Помилка: {e}")
        # Якщо не вийшло, повертаємося до базової сили (NewtonianAttraction)
        gravity = NewtonianAttraction(MU)
        propagator_builder.addForceModel(gravity)


    # Отримання об'єктів небесних тіл
    sun = CelestialBodyFactory.getSun()
    moon = CelestialBodyFactory.getMoon()

    # Створення моделей сил
    SunAttraction = ThirdBodyAttraction(sun)
    MoonAttraction = ThirdBodyAttraction(moon)

    # Додавання до пропагатора
    propagator_builder.addForceModel(SunAttraction)
    propagator_builder.addForceModel(MoonAttraction)


    #  Створення моделі Землі (OneAxisEllipsoid) для моделювання тіні
    # Використовуємо ITRF, велику піввісь та сплющеність WGS84
    earth_ellipsoid = OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,  # Сплющеність WGS84
        ITRF
    )

    sc_model = IsotropicRadiationSingleCoefficient(2.0, 0.5)  # Площа (m^2), Cr (коефіцієнт відбиття)

    # Тепер аргументи відповідають: (ExtendedPositionProvider, OneAxisEllipsoid, RadiationSensitive)
    SRP = SolarRadiationPressure(
        CelestialBodyFactory.getSun(),  # ExtendedPositionProvider (об'єкт Сонця)
        earth_ellipsoid,  # OneAxisEllipsoid (модель Землі)
        sc_model  # RadiationSensitive (модель апарату)
    )
    propagator_builder.addForceModel(SRP)


    print('Going into propagator...')
    # 3. Додаємо Будівельник до масиву (оскільки конструктор очікує масив)
    # Потрібно імпортувати JArray.
    # builder_array = JArray(KeplerianPropagatorBuilder)([propagator_builder])
    # builder_array = JArray(JClass("org.orekit.propagation.PropagatorBuilder"))([propagator_builder])
    # builder_array = JArray(JClass("org.orekit.propagation.AbstractPropagatorBuilder"))([propagator_builder])
    builder_array = JArray(JClass("org.orekit.propagation.conversion.PropagatorBuilder"))([propagator_builder])

    # Оптимізатор і оцінювач
    optimizer = LevenbergMarquardtOptimizer(
        1.0e-3,  # 1. initialStepBoundFactor (Зазвичай 1.0)
        1.0e-10,  # 2. costRelativeTolerance (Мала толерантність)
        1.0e-10,  # 3. parametersRelativeTolerance (Мала толерантність)
        1.0e-10,  # 4. costAbsoluteTolerance (Мала толерантність)
        1.0e-10  # 5. parametersAbsoluteTolerance (Мала толерантність)
    )
    # estimator = BatchLSEstimator(optimizer, propagator_builder)
    estimator = BatchLSEstimator(optimizer, builder_array)
    estimator.setMaxIterations(5000)  # Наприклад, 50 ітерацій
    estimator.setMaxEvaluations(10000)  # Наприклад, 100 обчислень (завжди більше, ніж ітерацій)

    sigma_angular = radians(20.0 / 3600.0)  # 1 кутова секунда в радіанах
    # print(sigma_angular)
    base_weight = 1.0

    for t, ra, dec in zip(times, ras, decs):
        date = datetime_to_absolutedate(t)
        observed_value = JArray(JDouble, 1)(np.array([ra, dec]))
        sigma_array = JArray(JDouble, 1)(np.array([sigma_angular, sigma_angular]))
        weight_array = JArray(JDouble, 1)(np.array([base_weight, base_weight]))

        meas = AngularRaDec(
            station, EME2000, date,
            observed_value, sigma_array, weight_array, satellite
        )
        estimator.addMeasurement(meas)

    # Оцінка орбіти
    estimated_propagator = estimator.estimate()
    # print("Propagator:", estimated_propagator)
    estimated_state = estimated_propagator[0].getInitialState()
    estimated_orbit = estimated_state.getOrbit()

    pv = estimated_orbit.getPVCoordinates()
    r = np.array([pv.getPosition().getX(), pv.getPosition().getY(), pv.getPosition().getZ()])
    v = np.array([pv.getVelocity().getX(), pv.getVelocity().getY(), pv.getVelocity().getZ()])

    kep = OrbitType.KEPLERIAN.convertType(estimated_orbit)

    # a_meters = estimated_orbit.getA()
    # mu = estimated_orbit.getMu()
    # # print(mu, a_meters)
    # # Обчислюємо середній рух (n) у радіанах/секунду за формулою: n = sqrt(mu / a^3)
    # # np має бути імпортовано (import numpy as np)
    # n_rad_per_sec = np.sqrt(mu / a_meters ** 3)
    # # Конвертація в оберти/добу для TLE
    # mm = n_rad_per_sec * 86400.0 / (2.0 * np.pi)
    # # print(mean_motion)


    elements = {
        "a": kep.getA()/1000,
        "e": kep.getE(),
        "i": np.degrees(kep.getI()),
        "raan": np.degrees(kep.getRightAscensionOfAscendingNode()),
        "argp": np.degrees(kep.getPerigeeArgument()),
        "M": np.degrees(kep.getMeanAnomaly())
    }

    tle = None
    print(elements)
    if make_tle:
        if elements['e'] >= 0.6:
            print(f"⚠️ Увага: Ексцентриситет занадто великий (e={elements['e']:.3f}) для TLE, TLE не згенеровано.")
        else:
            tle = make_tle_orekit(
                elements["a"], elements["e"], np.radians(elements["i"]),
                np.radians(elements["raan"]), np.radians(elements["argp"]),
                np.radians(elements["M"]),
                # mean_motion_tle=mm,
                norad=norad, cospar=cospar, abs_date=date #times[0] #.to_datetime().timestamp() / 86400.0 + 2440587.5
            )

    return {"r": r, "v": v, "elements": elements, "tle": tle}


def make_tle_orekit(akm, e, i, raan, argp, M, norad, cospar, abs_date):
    """Створення TLE через Orekit"""
    # print(ak, e, i, raan, argp, M, norad, cospar, epoch_jd)
    # print(type(ak), type(e), type(i), type(raan), type(argp), type(M), type(norad), type(cospar), type(epoch_jd))
    TLE = JClass("org.orekit.propagation.analytical.tle.TLE")
    TLEPropagator = JClass("org.orekit.propagation.analytical.tle.TLEPropagator")
    FramesFactory = JClass("org.orekit.frames.FramesFactory")
    Constants = JClass("org.orekit.utils.Constants")

    frame = FramesFactory.getTEME()
    # date = datetime_to_absolutedate(epoch_jd)

    aa = akm * 1000
    mean_motion_tle_rev = np.sqrt(Constants.WGS84_EARTH_MU / aa ** 3) * 86400.0 / (2 * np.pi)


    # =========================================================
    # 🌟 ПАРСИНГ COSPAR НОМЕРА
    # COSPAR: YYNNNL, наприклад, '04022A'
    # =========================================================
    try:
        # Рік запуску: 04 -> 2004
        launchYear = int(cospar[0:2])
        # Порядковий номер: 022 -> 22
        launchNumber = int(cospar[2:5])
        # Частина запуску: A
        launchPiece = str(cospar[5:]).strip().upper()
    except (IndexError, ValueError):
        print(f"⚠️ Помилка парсингу COSPAR '{cospar}'. Використовуються заглушки.")
        launchYear = 0
        launchNumber = 0
        launchPiece = str(cospar)  # Залишаємо весь рядок як launchPiece

    # Заглушки для інших TLE параметрів:
    ephemerisType = 0
    elementNumber = 999
    meanMotionFirstDerivative = 0.0
    meanMotionSecondDerivative = 0.0
    revolutionNumber = 0
    bStar = 0.0
    # =========================================================

    # cosparId_str = f"{launchYear}{launchNumber:03d}{launchPiece}"
    # print(
    #     launchYear,  # 3. launchYear (int) - З COSPAR
    #     launchNumber,  # 4. launchNumber (int) - З COSPAR
    #     launchPiece,  # 5. launchPiece (String) - З COSPAR
    #     ephemerisType,  # 6. ephemerisType (int)
    #     elementNumber,  # 7. elementNumber (int)
    #               )
    # Використовуємо 18-аргументний конструктор TLE:

    print('mean_motion=', mean_motion_tle_rev, type(mean_motion_tle_rev))
    tle = TLE(
        int(norad),  # 1. satelliteNumber (int)
        'U'[0],  # 2. classification (char)
        int(launchYear),  # 3. launchYear (int) - З COSPAR
        int(launchNumber),  # 4. launchNumber (int) - З COSPAR
        str(launchPiece),  # 5. launchPiece (String) - З COSPAR
        int(ephemerisType),  # 6. ephemerisType (int)
        int(elementNumber),  # 7. elementNumber (int)
        abs_date,  # 8. epoch (AbsoluteDate)
        float(mean_motion_tle_rev),  # 9. meanMotion (double)
        float(meanMotionFirstDerivative),  # 10. meanMotionFirstDerivative (double)
        float(meanMotionSecondDerivative),  # 11. meanMotionSecondDerivative (double)
        float(e),  # 12. eccentricity (double)
        float(np.degrees(i)),  # 13. inclination (double)
        float(np.degrees(raan)),  # 14. raan (double)
        float(np.degrees(argp)),  # 15. argPerigee (double)
        float(np.degrees(M)),  # 16. meanAnomaly (double)
        int(revolutionNumber),  # 17. revolutionNumber (int)
        float(bStar)  # 18. bStar (double)
    )

    # tle = TLE(
    #     int(norad), 'U', cospar,
    #     date, mean_motion, 0.0, 0.0,
    #     e, np.degrees(i), np.degrees(raan),
    #     np.degrees(argp), np.degrees(M), 0.0, 0
    # )

    prop = TLEPropagator.selectExtrapolator(tle)
    pv = prop.getPVCoordinates(abs_date, frame)

    return {
        "r": np.array([pv.getPosition().getX(), pv.getPosition().getY(), pv.getPosition().getZ()]),
        "v": np.array([pv.getVelocity().getX(), pv.getVelocity().getY(), pv.getVelocity().getZ()]),
        "tle": tle.toString()
    }
