import math

def distance_coords(m1, m2, r=6371):
    '''Calculate distance beetween coordinates (in km)'''

    phi1, lam1 = m1
    phi2, lam2 = m2

    phi1 = math.radians(phi1)
    phi2 = math.radians(phi2)
    lam1 = math.radians(lam1)
    lam2 = math.radians(lam2)

    return 2 * r * math.asin(
        math.sqrt(math.sin((phi2 - phi1) / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin((lam2 - lam1) / 2) ** 2))