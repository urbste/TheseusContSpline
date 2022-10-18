
S_TO_NS = 1e9
NS_TO_S = 1e-9

def calc_times(sensor_time_ns, start_time_ns, dt_ns, nr_knots, N):
    st_ns = sensor_time_ns - start_time_ns

    u = 0.0

    if st_ns < 0.0:
        return 0.0, 0, False

    
    s = int(st_ns / dt_ns)
    if s < 0:
        return 0.0, 0, False
    if s + N > nr_knots:
        return u, s, False

    u = (st_ns % dt_ns) / dt_ns

    return u, s, True

