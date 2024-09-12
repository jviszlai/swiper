from swiper.surfaces import surface_patch, lattice_surgery_op

data_q = [surface_patch((i, 1)) for i in range(5)]
lower_routing_q = [surface_patch((i, 0)) for i in range(5)]
upper_routing_q = [surface_patch((i, 2)) for i in range(5)]

def msd_op_list(t0:int = 0, delays=0):
    return [
        lattice_surgery_op(data_q, [data_q[1]], [upper_routing_q[1]]).gen_subgraph(t0=t0, delays=delays),
        lattice_surgery_op(data_q, [data_q[2]], [upper_routing_q[2]]).gen_subgraph(t0=t0, delays=delays),
        lattice_surgery_op(data_q, [data_q[3]], [upper_routing_q[3]]).gen_subgraph(t0=t0, delays=delays),

        lattice_surgery_op(data_q, [data_q[i] for i in range(1,4)], [lower_routing_q[i] for i in range(1,4)]).gen_subgraph(t0=t0+(delays+1), delays=delays),
        lattice_surgery_op(data_q, [data_q[i] for i in range(3)], [upper_routing_q[i] for i in range(3)]).gen_subgraph(t0=t0+2*(delays+1), delays=delays),
        lattice_surgery_op(data_q, [data_q[i] for i in range(2)] + [data_q[3]], [lower_routing_q[i] for i in range(4)]).gen_subgraph(t0=t0+3*(delays+1), delays=delays),

        lattice_surgery_op(data_q, [data_q[4]], [upper_routing_q[4]]).gen_subgraph(t0=t0+4*(delays+1), delays=delays),
        lattice_surgery_op(data_q, [data_q[0]] + [data_q[i] for i in range(2,4)], [upper_routing_q[i] for i in range(4)]).gen_subgraph(t0=t0+4*(delays+1), delays=delays),
        lattice_surgery_op(data_q, [data_q[0]] + [data_q[i] for i in range(3,5)], [lower_routing_q[i] for i in range(5)]).gen_subgraph(t0=t0+5*(delays+1), delays=delays),

        lattice_surgery_op(data_q, [data_q[i] for i in range(2)] + [data_q[4]], [upper_routing_q[i] for i in range(5)]).gen_subgraph(t0=t0+6*(delays+1), delays=delays),
        lattice_surgery_op(data_q, [data_q[i] for i in range(0,5,2)], [lower_routing_q[i] for i in range(5)]).gen_subgraph(t0=t0+7*(delays+1), delays=delays),

        lattice_surgery_op(data_q, [data_q[i] for i in range(5)], [upper_routing_q[i] for i in range(5)]).gen_subgraph(t0=t0+8*(delays+1), delays=delays),
        lattice_surgery_op(data_q, [data_q[i] for i in range(2,5)], [lower_routing_q[i] for i in range(2,5)]).gen_subgraph(t0=t0+9*(delays+1), delays=delays),
        
        lattice_surgery_op(data_q, [data_q[1]] + [data_q[i] for i in range(3,5)], [upper_routing_q[i] for i in range(1,5)]).gen_subgraph(t0=t0+10*(delays+1), delays=delays),
        lattice_surgery_op(data_q, [data_q[i] for i in range(1,3)] + [data_q[4]], [lower_routing_q[i] for i in range(1,5)]).gen_subgraph(t0=t0+11*(delays+1), delays=delays),
    ]

