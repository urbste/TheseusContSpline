

def calc_times(sensor_time_ns, start_time_ns, dt_ns, nr_knots, N):
    st_ns = sensor_time_ns - start_time_ns

    if st_ns < 0.0:
        u = 0.0
        return 0.0, 0, False

    
    s = st_ns / dt_ns
    if s < 0:
        return 0.0, 0, False
    if s + N > nr_knots:
        return u, s, False

    u = (st_ns % dt_ns) / dt_ns

    return u, s, True

