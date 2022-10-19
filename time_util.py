import torch

S_TO_NS = 1e9
NS_TO_S = 1e-9

def calc_times(sensor_time_ns, start_time_ns, dt_ns, nr_knots, N):
    st_ns = sensor_time_ns - start_time_ns

    u = torch.tensor(0.0, dtype=torch.float32)
    s = torch.tensor(0, dtype=torch.int)

    if st_ns < 0.0:
        return u, s, False

    
    s = torch.tensor(st_ns / dt_ns, dtype=torch.int)

    if s < 0:
        return u, torch.tensor(0, dtype=torch.int), False

    if s + N > nr_knots:
        return u, s, False

    u = (st_ns % dt_ns) / dt_ns

    return u, s, True

