# tests/test_laplace.py — остаточна робоча версія

from astropy.time import Time
from methods_laplace_new2 import laplace_od   # або просто from methods_laplace import laplace_od


def test_laplace_geo_like():
    times = Time([
        "2001-05-23T20:16:24.024",
        "2001-05-23T20:36:48.938",
        "2001-05-23T20:57:13.852"
    ], scale="utc")

    ra_deg = [283.92195, 294.67483, 303.82137]
    dec_deg = [-34.38700, -41.46583, -45.36542]

    obs = (times, ra_deg, dec_deg, None, None, None)

    result = laplace_od(obs, lat=19.0, lon=-155.0, h=0.0, dbg=False)

    el = result["elements"]
    print(el)

    assert 7000 < el["a"] < 7300
    assert el["e"] < 0.01
    assert 25 < el["i"] < 32