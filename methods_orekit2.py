import os
import sys
import numpy as np
from math import radians
import jpype.imports
from jpype import JClass, JArray, JDouble, getDefaultJVMPath



# ================================================================
# 1️⃣  ІНІЦІАЛІЗАЦІЯ JVM І ЗАВАНТАЖЕННЯ OREKIT
# ================================================================
def init_jvm_orekit(orekit_dir="./orekit_lib", data_dir="orekit-data"):
    """
    Ініціалізує JVM з Orekit та Hipparchus JAR-файлів.
    Викликається один раз перед роботою з Orekit через JPype.

    orekit_dir : шлях до директорії, де лежать orekit*.jar та hipparchus*.jar
    data_dir   : шлях до папки з orekit-data
    """
    if jpype.isJVMStarted():
        print("ℹ️ JVM уже запущено, ініціалізація пропущена.")
        return

    # Пошук JAR-файлів у каталозі
    if not os.path.exists(orekit_dir):
        raise FileNotFoundError(f"Каталог з JAR-файлами не знайдено: {orekit_dir}")

    jar_files = [os.path.join(orekit_dir, f) for f in os.listdir(orekit_dir) if f.endswith(".jar")]
    if not jar_files:
        raise RuntimeError(f"У каталозі {orekit_dir} не знайдено жодного .jar файлу")

    # Формуємо classpath
    classpath_sep = ";" if sys.platform.startswith("win") else ":"
    classpath = classpath_sep.join(jar_files)

    jvm_path = getDefaultJVMPath()
    print(f"🟢 Використовується JVM: {jvm_path}")
    print(f"🟢 Завантаження JAR-файлів з: {orekit_dir}")

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
    manager.addProvider(DirectoryCrawler(File(data_dir)))
    # print(manager)
    print(f"✅ Orekit data loaded successfully from: {data_dir}")







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
        a = EARTH_RADIUS + 700e3
        v = np.sqrt(MU / a)
        pv = PVCoordinates(Vector3D(float(a), 0.0, 0.0), Vector3D(0.0, float(v), 0.0))
        initial_state = CartesianOrbit(pv, TEME, date, MU)
    else:
        r2 = initial_state["r"]  # Вектор позиції (m)
        v2 = initial_state["v"]  # Вектор швидкості (m/s)

        # Перетворення списків/масивів у Vector3D (припускаючи, що r2 і v2 - це масиви/списки з 3 елементів)
        position = Vector3D(float(r2[0]), float(r2[1]), float(r2[2]))
        velocity = Vector3D(float(v2[0]), float(v2[1]), float(v2[2]))

        pv = PVCoordinates(position, velocity)
        date = datetime_to_absolutedate(times[0])
        initial_state = CartesianOrbit(pv, TEME, date, MU)
        # elements = initial_state["elements"]
        #
        # # Перетворення елементів: кути з градусів у радіани
        # a = elements["a"]  # Велика піввісь (м)
        # e = elements["e"]  # Ексцентриситет
        # i = np.radians(elements["i"])  # Нахил (радіани)
        # raan = np.radians(elements["raan"])  # Довгота висхідного вузла (радіани)
        # argp = np.radians(elements["argp"])  # Аргумент перицентру (радіани)
        # nu = np.radians(elements["nu"])  # Справжня аномалія (радіани)
        #
        # # Створення KeplerianOrbit
        # initial_state = KeplerianOrbit(
        #     a, e, i, raan, argp, nu,
        #     PositionAngleType.TRUE,  # Вказуємо, що nu - це Справжня аномалія
        #     TEME, date, MU
        # )

    # # Побудова пропагатора
    # propagator_builder = NumericalPropagatorBuilder(
    #     initial_state,
    #     DormandPrince853IntegratorBuilder(1.0, 300.0, 1.0e-3),
    #     PositionAngleType.TRUE,
    #     1.0
    # )
    # gravity = NewtonianAttraction(MU)
    # propagator_builder.addForceModel(gravity)

    # Визначте константи J2
    J2 = Constants.WGS84_EARTH_C20 * -np.sqrt(5)  # C20 * (-sqrt(5)) = J2 (для Orekit)

    # Імпорт потрібних класів
    HolmesFeatherstoneAttractionModel = JClass("org.orekit.forces.gravity.HolmesFeatherstoneAttractionModel")
    GravityFieldFactory = JClass(
        "org.orekit.forces.gravity.potential.GravityFieldFactory")  # Цей клас потрібен для отримання потенціалу

    # Додайте до секції імпортів:
    SphericalHarmonicsProvider = JClass("org.orekit.forces.gravity.potential.SphericalHarmonicsProvider")
    # GravityFieldFactory = JClass("org.orekit.forces.gravity.potential.GravityFieldFactory")

    # Створення моделі гравітаційного поля (наприклад, WGS84 EGM)
    # gravity_model = HolmesFeatherstone(
    #     ITRF,
    #     Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
    #     MU,
    #     # Матриця коефіцієнтів C (1 рядок для J2, оскільки J2 = -C20 * sqrt(5))
    #     JArray(JDouble, 2)([JArray(JDouble, 2)([0.0, 0.0]), JArray(JDouble, 2)([0.0, -J2])]),
    #     # Матриця коефіцієнтів S (просто нулі)
    #     JArray(JDouble, 2)([JArray(JDouble, 2)([0.0, 0.0]), JArray(JDouble, 2)([0.0, 0.0])])
    # )

    # propagator_builder = KeplerianPropagatorBuilder(
    #     initial_state,  # Початкова оцінка
    #     PositionAngleType.MEAN,
    #     1.0,  # Sigma, похибка моделі (1.0 - стандартне значення)
    # )

    integrator_builder = DormandPrince853IntegratorBuilder(1.0, 300.0, 1.0e-3)
    # ❗️ Використовуйте NumericalPropagatorBuilder:
    propagator_builder = NumericalPropagatorBuilder(
        initial_state,
        integrator_builder,
        # OrbitType.CARTESIAN,  # Більш стійкий тип орбіти для чисельного інтегрування
        PositionAngleType.TRUE,
        1.0  # Sigma
    )

    try:
        # Завантаження стандартного гравітаційного поля (наприклад, WGS84 EGM)
        # З порядком і ступенем (degree and order) 2 - це J2.
        # Використовуємо 5x5, щоб мати трохи більше точності.
        # IERS_2010 гарантує правильні константи.
        provider = GravityFieldFactory.getConstantNormalizedProvider(
            5, 5, IERSConventions.IERS_2010, True
        )

        # Створюємо ForceModel на основі потенціалу
        force_model = HolmesFeatherstoneAttractionModel(ITRF, provider)

        propagator_builder.addForceModel(force_model)
    except Exception as e:
        print(
            f"⚠️ Помилка при додаванні моделі J2 (гравітаційний потенціал). Спробуйте лише NewtonianAttraction. Помилка: {e}")
        # Якщо не вийшло, повертаємося до базової сили (NewtonianAttraction)
        gravity = NewtonianAttraction(MU)
        propagator_builder.addForceModel(gravity)

    # 3. Додаємо Будівельник до масиву (оскільки конструктор очікує масив)
    # Потрібно імпортувати JArray.
    # builder_array = JArray(KeplerianPropagatorBuilder)([propagator_builder])
    # builder_array = JArray(JClass("org.orekit.propagation.PropagatorBuilder"))([propagator_builder])
    # builder_array = JArray(JClass("org.orekit.propagation.AbstractPropagatorBuilder"))([propagator_builder])
    builder_array = JArray(JClass("org.orekit.propagation.conversion.PropagatorBuilder"))([propagator_builder])

    # Оптимізатор і оцінювач
    optimizer = LevenbergMarquardtOptimizer()
    # estimator = BatchLSEstimator(optimizer, propagator_builder)
    estimator = BatchLSEstimator(optimizer, builder_array)
    estimator.setMaxIterations(5000)  # Наприклад, 50 ітерацій
    estimator.setMaxEvaluations(10000)  # Наприклад, 100 обчислень (завжди більше, ніж ітерацій)

    sigma_angular = radians(1.0 / 3600.0)  # 1 кутова секунда в радіанах
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

    elements = {
        "a": kep.getA(),
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
                norad, cospar, times[0] #.to_datetime().timestamp() / 86400.0 + 2440587.5
            )

    return {"r": r, "v": v, "elements": elements, "tle": tle}


def make_tle_orekit(a, e, i, raan, argp, M, norad, cospar, epoch_jd):
    """Створення TLE через Orekit"""
    TLE = JClass("org.orekit.propagation.analytical.tle.TLE")
    TLEPropagator = JClass("org.orekit.propagation.analytical.tle.TLEPropagator")
    FramesFactory = JClass("org.orekit.frames.FramesFactory")
    Constants = JClass("org.orekit.utils.Constants")

    frame = FramesFactory.getTEME()
    date = datetime_to_absolutedate(epoch_jd)

    mean_motion = np.sqrt(Constants.WGS84_EARTH_MU / a ** 3) * 86400.0 / (2 * np.pi)

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

    # Використовуємо 18-аргументний конструктор TLE:
    tle = TLE(
        int(norad),  # 1. satelliteNumber (int)
        'U'[0],  # 2. classification (char)
        launchYear,  # 3. launchYear (int) - З COSPAR
        launchNumber,  # 4. launchNumber (int) - З COSPAR
        launchPiece,  # 5. launchPiece (String) - З COSPAR
        ephemerisType,  # 6. ephemerisType (int)
        elementNumber,  # 7. elementNumber (int)
        date,  # 8. epoch (AbsoluteDate)
        float(mean_motion),  # 9. meanMotion (double)
        meanMotionFirstDerivative,  # 10. meanMotionFirstDerivative (double)
        meanMotionSecondDerivative,  # 11. meanMotionSecondDerivative (double)
        float(e),  # 12. eccentricity (double)
        float(np.degrees(i)),  # 13. inclination (double)
        float(np.degrees(raan)),  # 14. raan (double)
        float(np.degrees(argp)),  # 15. argPerigee (double)
        float(np.degrees(M)),  # 16. meanAnomaly (double)
        revolutionNumber,  # 17. revolutionNumber (int)
        bStar  # 18. bStar (double)
    )

    # tle = TLE(
    #     int(norad), 'U', cospar,
    #     date, mean_motion, 0.0, 0.0,
    #     e, np.degrees(i), np.degrees(raan),
    #     np.degrees(argp), np.degrees(M), 0.0, 0
    # )

    prop = TLEPropagator.selectExtrapolator(tle)
    pv = prop.getPVCoordinates(date, frame)

    return {
        "r": np.array([pv.getPosition().getX(), pv.getPosition().getY(), pv.getPosition().getZ()]),
        "v": np.array([pv.getVelocity().getX(), pv.getVelocity().getY(), pv.getVelocity().getZ()]),
        "tle": tle.toString()
    }
