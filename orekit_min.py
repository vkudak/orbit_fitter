import numpy as np
import datetime as dt
from datetime import timezone
from math import radians
import os
import sys


import jpype
import jpype.imports
from jpype import JClass, getDefaultJVMPath


# orekit_jar = "/home/vkudak/orekit_lib/orekit-13.1.jar"
# hip_jar = "/home/vkudak/orekit_lib/hipparchus-core-4.0.2.jar"
# hip_geom_jar = "/home/vkudak/orekit_lib/hipparchus-geometry-4.0.2.jar"


orekit_dir = "/home/vkudak/orekit_lib/"  # директорія з orekit + hipparchus JAR

# Створюємо список усіх JAR-файлів у директорії
jar_files = [os.path.join(orekit_dir, f) for f in os.listdir(orekit_dir) if f.endswith(".jar")]

# Об'єднуємо в рядок для classpath (Linux: ':', Windows: ';')
classpath = ":".join(jar_files)
print(classpath)

jvm_path = getDefaultJVMPath()

print("Using JVM:", jvm_path)
# print("Using Orekit JAR:", orekit_jar)

# Запуск JVM із ключами для сучасних JVM
jpype.startJVM(
    jvm_path,
    "-ea",
    "--enable-native-access=ALL-UNNAMED",
    f"-Djava.class.path={classpath}" #{orekit_jar}:{hip_jar}:{hip_geom_jar}"
)

# # Імпорти класів
# File = JClass("java.io.File")
# DirectoryCrawler = JClass("org.orekit.data.DirectoryCrawler")
# DataProvidersManager = JClass("org.orekit.data.DataProvidersManager")
#
# # 🔹 Створюємо новий менеджер явно
# manager = DataProvidersManager()
# print("DataProvidersManager created:", manager)
#
# # Завантаження даних
# data_path = "orekit-data"
# data_dir = File(data_path)
# manager.clearProviders()
# manager.addProvider(DirectoryCrawler(data_dir))
# print("Orekit data loaded from:", data_path)
#
# # Перевірка констант
# Constants = JClass("org.orekit.utils.Constants")
# print("Earth mu:", Constants.WGS84_EARTH_MU)

from org.orekit.data import DataContext, DirectoryCrawler
from java.io import File

data_dir = "orekit-data"
manager = DataContext.getDefault().getDataProvidersManager()
manager.addProvider(DirectoryCrawler(File(data_dir)))



from org.orekit.time import AbsoluteDate, TimeScalesFactory

def datetime_to_absolutedate(dt_utc: dt.datetime) -> 'AbsoluteDate':
    """Перетворення Python datetime UTC в Orekit AbsoluteDate"""
    utc = TimeScalesFactory.getUTC()
    return AbsoluteDate(
        dt_utc.year,
        dt_utc.month,
        dt_utc.day,
        dt_utc.hour,
        dt_utc.minute,
        dt_utc.second + dt_utc.microsecond / 1e6,
        utc
    )


# jpype.shutdownJVM()
# sys.exit()
# --- 1. Ініціалізація Orekit та завантаження даних ---
# import orekit
# from orekit.pyhelpers import datetime_to_absolutedate

# # --- 1. Запуск JVM до імпорту JPype ---
# vm = orekit.initVM()  # обов'язково до будь-якого імпорту з jpype
#
# # --- 2. Тільки тепер можна імпортувати класи з jpype ---
#
# from jpype import JArray, JDouble
# import jpype
#
# import jpype
# from jpype import getDefaultJVMPath, isJVMStarted, startJVM
#
# jvm_path = getDefaultJVMPath()
# print("Using JVM:", jvm_path)
#
# # Ключовий момент — прапорець --enable-native-access
# startJVM(
#     jvm_path,
#     "-ea",
#     "--enable-native-access=ALL-UNNAMED",
#     "-Djava.library.path=/home/vkudak/miniconda3/envs/orbit_fitter/jre/lib/amd64/server"
# )
# print("JVM started:", isJVMStarted())
# jpype.shutdownJVM()

# -----------------------------------

# # Ініціалізація JVM (це має бути першим)
# try:
#     orekit.initVM()
# except RuntimeError:
#     pass  # JVM вже ініціалізовано
# --- ІНІЦІАЛІЗАЦІЯ JVM (ПОВТОРНО) ---
# # Переконайтеся, що ви виконуєте ініціалізацію лише один раз
# try:
#     orekit.initVM()
#     # Якщо тут трапилася помилка, вона буде відображена.
# except RuntimeError:
#     # Цей блок спрацьовує, якщо JVM вже була запущена
#     pass
# except Exception as e:
#     # Якщо initVM() дав збій через конфлікт або відсутність Java
#     print(f"❌ Критична помилка запуску JVM під час initVM(): {e}")
# ------------------------------------

# # ✅ ВСТАВИТИ ЯВНИЙ ЗАПУСК JPYPE:
# if not jpype.isJVMStarted():
#     try:
#         # ЗМІНІТЬ ЦЕЙ ШЛЯХ НА ВАШ:
#         OREKIT_JAR = '/home/vkudak/miniconda3/envs/orbit_fitter/lib/python3.10/site-packages/orekit/orekit.jar'
#
#         jpype.startJVM(
#             jpype.getDefaultJVMPath(),
#             '-ea',
#             f'-Djava.class.path={OREKIT_JAR}'
#         )
#         print("✅ JPype JVM успішно запущено вручну.")
#     except Exception as e:
#         print(f"❌ Критична помилка запуску JPype: {e}")


# --- 2. ІМПОРТ КЛАСІВ OREKIT/JAVA ПІСЛЯ initVM() ---
# Ці класи Java-залежні і вимагають запущеної JVM.
from java.io import File
from org.orekit.data import DirectoryCrawler, ZipJarCrawler, DataContext
from org.orekit.orbits import CartesianOrbit, PositionAngleType, OrbitType
from org.orekit.frames import FramesFactory
from org.orekit.time import TimeScalesFactory
from org.orekit.utils import Constants, IERSConventions, PVCoordinates
from org.orekit.estimation.leastsquares import BatchLSEstimator
from org.orekit.estimation.measurements import AngularRaDec, ObservableSatellite, GroundStation
from org.orekit.propagation.conversion import NumericalPropagatorBuilder, DormandPrince853IntegratorBuilder
from org.orekit.forces.gravity import NewtonianAttraction
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.hipparchus.optim.nonlinear.vector.leastsquares import LevenbergMarquardtOptimizer
from org.orekit.frames import TopocentricFrame
from org.orekit.bodies import OneAxisEllipsoid, GeodeticPoint
from jpype import JArray, JDouble  # ✅ JPype імпортуємо тут
# # --- 3. ЗАВАНТАЖЕННЯ ДАНИХ OREKIT (ВИКОРИСТАННЯ КЛАСІВ) ---
# data_path = "orekit-data"
#
# # 2. Налаштування менеджера даних
# manager = DataContext.getDefault().getDataProvidersManager()
# file = File(data_path)
#
# if file.isDirectory():
#     crawler = DirectoryCrawler(file)
# elif data_path.lower().endswith('.zip'):
#     crawler = ZipJarCrawler(file)
# else:
#     raise FileNotFoundError(f"Дані Orekit не знайдені за шляхом: {data_path}")
# manager.addProvider(crawler)

# Глобальні константи
ITRF = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
TEME = FramesFactory.getTEME()
UTC = TimeScalesFactory.getUTC()
MU = Constants.WGS84_EARTH_MU
EARTH_RADIUS = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
EME2000 = FramesFactory.getEME2000()


def orbit_determination_from_radec(times: np.ndarray, ras: np.ndarray, decs: np.ndarray, station_lla: tuple,
                                   initial_orbit_guess: CartesianOrbit) -> CartesianOrbit:
    """
    Виконує визначення орбіти (Orbit Determination) з вимірювань пряме сходження/схилення.
    """

    # --- 2. Налаштування Середовища ---
    # Земля
    earth = OneAxisEllipsoid(EARTH_RADIUS, Constants.WGS84_EARTH_FLATTENING, ITRF)

    # Станція спостереження (широта, довгота, висота - в радіанах та метрах)
    station_latitude, station_longitude, station_altitude = station_lla

    # Отримуємо вектор положення станції в ITRF з геодетичних координат
    geo_point = GeodeticPoint(station_latitude, station_longitude, station_altitude)

    # 2. Використовуємо метод transform, щоб отримати Vector3D в ITRF (Earth Frame)
    # Перетворюємо геодетичні координати (geo_point) в декартові (Vector3D) у фреймі ITRF
    # point = earth.transform(geo_point)

    # Створюємо TopocentricFrame
    station_frame = TopocentricFrame(
        earth,
        geo_point,
        "MyStation"
    )
    station = GroundStation(station_frame)
    satellite = ObservableSatellite(0)

    # Налаштування моделі сил (Пропагатор)
    propagator_builder = NumericalPropagatorBuilder(
        initial_orbit_guess,
        DormandPrince853IntegratorBuilder(1.0, 300.0, 1.0e-3),
        PositionAngleType.TRUE,
        1.0
    )

    # Додавання сили гравітації (4x4)
    # gravity = (HolmesFeatherstone(
    #     ITRF,
    #     GravityFieldFactory.getConstantNormalizedHarmonicsProvider(4, 4)
    # ))

    gravity = NewtonianAttraction(MU)
    propagator_builder.addForceModel(gravity)

    # --- 3. Налаштування Оцінювача ---
    # optimizer = LevenbergMarquardtOptimizer()
    # Налаштування параметрів (використовуємо Python float для double):
    convergence_threshold = 1.0e-3
    cost_threshold = 1.0e-6
    initial_lambda = 1.0e-1
    final_lambda = 1.0e+6
    # Максимальна кількість обчислень (наприклад, 1000)
    max_evaluations = 1000

    optimizer = LevenbergMarquardtOptimizer(
        convergence_threshold,
        cost_threshold,
        initial_lambda,
        final_lambda,
        JDouble(max_evaluations)  # Обов'язково явно передаємо maxEvaluations як double
    )

    estimator = BatchLSEstimator(optimizer, propagator_builder)

    # --- 4. Створення Вимірювань ---
    if len(times) != len(ras) or len(times) != len(decs):
        raise ValueError("Розміри масивів times, ras та decs мають збігатися.")

    sigma_angular = radians(1.0 / 3600.0)  # 1 кутова секунда в радіанах
    base_weight = 1.0


    for time_val, ra_val, dec_val in zip(times, ras, decs):
        date = datetime_to_absolutedate(time_val)

        # angular_measurement = AngularRaDec(
        #     station,
        #     date,
        #     np.array([ra_val, dec_val]),
        #     np.array([sigma_angular, sigma_angular]),
        #     np.array([base_weight, base_weight]),
        #     satellite
        # )
        # Створення Java-масивів Double для коректної передачі
        observed_value = JArray(JDouble, 1)(np.array([ra_val, dec_val]))
        sigma_array = JArray(JDouble, 1)(np.array([sigma_angular, sigma_angular]))
        weight_array = JArray(JDouble, 1)(np.array([base_weight, base_weight]))

        # observed_value_list = [ra_val, dec_val]
        # sigma_array_list = [sigma_angular, sigma_angular]
        # weight_array_list = [base_weight, base_weight]

        # # ЦЕ ЛИШЕ ДЛЯ ДІАГНОСТИКИ:
        # temp_array = np.array([ra_val, dec_val], dtype=np.float64)
        #
        # # Якщо у вашій збірці є вбудований метод перетворення:
        # try:
        #     java_double_array = temp_array.toDoubleArray()
        #     print("✅ toDoubleArray() доступний!")
        # except AttributeError:
        #     # Якщо він недоступний, ми знову повертаємося до того, що потрібно JArray/JDouble.
        #     print("❌ toDoubleArray() недоступний. Потрібен явний імпорт JPype.")

        # ✅ ВИПРАВЛЕНО: Передаємо Java-масиви
        angular_measurement = AngularRaDec(
            station,
            EME2000,
            date,
            observed_value,
            sigma_array,
            weight_array,
            satellite
        )


        estimator.addMeasurement(angular_measurement)

    # --- 5. Запуск Оцінювача ---
    estimated_propagator = estimator.estimate()

    # Отримання оціненої орбіти
    estimated_initial_state = estimated_propagator.getInitialState()
    estimated_orbit = estimated_initial_state.getOrbit()

    return estimated_orbit


# --- ПРИКЛАД ВИКОРИСТАННЯ (Заглушки) ---

# Фіктивні спостереження
start_time = dt.datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
time_step = dt.timedelta(minutes=10)
num_obs = 10
obs_times = np.array([start_time + i * time_step for i in range(num_obs)])

# Фіктивні дані RA/Dec (в радіанах)
obs_ras = np.array([radians(180 + i * 0.1) for i in range(num_obs)])
obs_decs = np.array([radians(10 + i * 0.05) for i in range(num_obs)])

obs = (obs_times, obs_ras, obs_decs)

# print(obs)

# Координати наземної станції (Широта, Довгота, Висота - в радіанах та метрах)
station_lla_rad = (radians(0.0), radians(0.0), 0.0)

# Початкове припущення орбіти (наприклад, кругова орбіта 500 км)
a = Constants.WGS84_EARTH_EQUATORIAL_RADIUS + 36000e3

# Константа швидкості (щоб уникнути помилки, якщо np.sqrt(MU/a) повертає Python float)
initial_velocity_y = np.sqrt(MU / a)

initial_pva = PVCoordinates(
    Vector3D(float(a), 0.0, 0.0),
    Vector3D(0.0, float(initial_velocity_y), 0.0)
)
initial_date = datetime_to_absolutedate(start_time)
initial_guess = CartesianOrbit(initial_pva, TEME, initial_date, MU)

print("Розпочато визначення орбіти...")
try:
    estimated_orbit = orbit_determination_from_radec(
        obs[0],
        obs[1],
        obs[2],
        station_lla_rad,
        initial_guess
    )

    print("\n✅ Оцінена орбіта (Estimated Orbit):")
    # Перетворення в елементи Кеплера
    keplerian_orbit = OrbitType.KEPLERIAN.convertType(estimated_orbit)

    print(f"  Епоха: {estimated_orbit.getDate()}")
    print(f"  Велика піввісь (a): {keplerian_orbit.getA():.2f} м")
    print(f"  Ексцентриситет (e): {keplerian_orbit.getE():.6f}")
    print(f"  Нахил (i): {np.degrees(keplerian_orbit.getI()):.4f}°")

except Exception as e:
    print(f"❌ Виникла помилка під час визначення орбіти: {e}")
    print(
        "Основні можливі причини: 1) Неправильно вказаний шлях до даних Orekit. 2) Початкове припущення занадто далеке від реальності.")