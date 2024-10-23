import numpy as np
from swiper2.lattice_surgery_schedule import LatticeSurgerySchedule, Duration

class MSD15To1Schedule:
    """Magic state distillation, according to arXiv:1808.06709v4. Uses a total
    spatial footprint of 4x8 patches. Output magic state is in location (0,0).

    Uses improvement from Fig. 8 of https://arxiv.org/pdf/1905.08916.
    """
    def __init__(self):
        """Builds the schedule.
        """
        schedule = LatticeSurgerySchedule()
        schedule.merge([(0,0), (0,1), (0,2), (0,3), (2,0), (2,1), (2,2), (2,4)], [(1,0), (1,1), (1,2), (1,3), (1,4)])
        schedule.merge([(0,0), (0,1), (0,4), (0,5), (2,0), (2,1), (2,3), (2,5)], [(1,0), (1,1), (1,2), (1,3), (1,4), (1,5)])
        schedule.merge([(0,0), (0,2), (0,4), (0,6), (2,0), (2,2), (2,3), (2,6)], [(1,0), (1,1), (1,2), (1,3), (1,4), (1,5), (1,6)])
        schedule.merge([(0,0), (0,3), (0,5), (0,6), (2,1), (2,2), (2,3), (2,7)], [(1,0), (1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7)])
        schedule.merge([(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7)], [(1,0), (1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7)])
        schedule.inject_T((1,0))
        schedule.inject_T((1,1))
        schedule.inject_T((1,2))
        schedule.inject_T((1,3))
        schedule.inject_T((1,4))
        schedule.inject_T((1,5))
        schedule.inject_T((1,6))
        schedule.inject_T((3,0))
        schedule.inject_T((3,1))
        schedule.inject_T((3,2))
        schedule.inject_T((3,3))
        schedule.inject_T((3,4))
        schedule.inject_T((3,5))
        schedule.inject_T((3,6))
        schedule.inject_T((3,7))
        schedule.merge([(1,0), (0,0)], [], duration=Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.merge([(1,1), (0,1)], [], duration=Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.merge([(1,2), (0,2)], [], duration=Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.merge([(1,3), (0,3)], [], duration=Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.merge([(1,4), (0,4)], [], duration=Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.merge([(1,5), (0,5)], [], duration=Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.merge([(1,6), (0,6)], [], duration=Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.merge([(3,0), (2,0)], [], duration=Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.merge([(3,1), (2,1)], [], duration=Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.merge([(3,2), (2,2)], [], duration=Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.merge([(3,3), (2,3)], [], duration=Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.merge([(3,4), (2,4)], [], duration=Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.merge([(3,5), (2,5)], [], duration=Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.merge([(3,6), (2,6)], [], duration=Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.merge([(3,7), (2,7)], [], duration=Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.discard([(1,0)])
        schedule.discard([(1,1)])
        schedule.discard([(1,2)])
        schedule.discard([(1,3)])
        schedule.discard([(1,4)])
        schedule.discard([(1,5)])
        schedule.discard([(1,6)])
        schedule.discard([(3,0)])
        schedule.discard([(3,1)])
        schedule.discard([(3,2)])
        schedule.discard([(3,3)])
        schedule.discard([(3,4)])
        schedule.discard([(3,5)])
        schedule.discard([(3,6)])
        schedule.discard([(3,7)])
        schedule.idle([(0,0)], Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.idle([(0,1)], Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.idle([(0,2)], Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.idle([(0,3)], Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.idle([(0,4)], Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.idle([(0,5)], Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.idle([(0,6)], Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.idle([(2,0)], Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.idle([(2,1)], Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.idle([(2,2)], Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.idle([(2,3)], Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.idle([(2,4)], Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.idle([(2,5)], Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.idle([(2,6)], Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.idle([(2,7)], Duration.HALF_D_ROUNDS_ROUNDED_DOWN)
        schedule.conditional_S((0,0), 25)
        schedule.conditional_S((0,1), 26)
        schedule.conditional_S((0,2), 27)
        schedule.conditional_S((0,3), 28)
        schedule.conditional_S((0,4), 29)
        schedule.conditional_S((0,5), 30)
        schedule.conditional_S((0,6), 31)
        schedule.conditional_S((2,0), 32)
        schedule.conditional_S((2,1), 33)
        schedule.conditional_S((2,2), 34)
        schedule.conditional_S((2,3), 35)
        schedule.conditional_S((2,4), 36)
        schedule.conditional_S((2,5), 37)
        schedule.conditional_S((2,6), 38)
        schedule.conditional_S((2,7), 39)
        schedule.discard([(0,0)])
        schedule.discard([(0,1)])
        schedule.discard([(0,2)])
        schedule.discard([(0,3)])
        schedule.discard([(0,4)])
        schedule.discard([(0,5)])
        schedule.discard([(0,6)])
        schedule.discard([(2,0)])
        schedule.discard([(2,1)])
        schedule.discard([(2,2)])
        schedule.discard([(2,3)])
        schedule.discard([(2,4)])
        schedule.discard([(2,5)])
        schedule.discard([(2,6)])
        schedule.discard([(2,7)])
        schedule.discard([(0,7)], conditioned_on_idx=set(range(65,80)))

        self.schedule = schedule

class MemorySchedule:
    """Simple schedule that idles one patch for a specified number of rounds.
    """
    def __init__(self, num_rounds: int):
        """Builds the schedule.

        Args:
            num_rounds: Number of rounds to idle.
        """
        schedule = LatticeSurgerySchedule()
        schedule.idle([(0,0)], num_rounds)
        schedule.discard([(0,0)])

        self.schedule = schedule

class RegularTSchedule:
    """Simple schedule that injects a T gate on one patch for a specified number
    of rounds, with a specified number of idle rounds between each T gate.
    Alternates T state creation between two adjacent patches.
    """
    def __init__(self, num_Ts: int, idle_between_Ts: int):
        """Builds the schedule.
        
        Args:
            num_Ts: Number of T gates to inject.
            idle_between_Ts: Number of idle rounds between each T gate.
        """
        schedule = LatticeSurgerySchedule()
        prev_injection_flag = False
        for i in range(num_Ts):
            schedule.idle([(0,0)], idle_between_Ts)

            injection_patch = (0,1) if prev_injection_flag else (1,0)
            schedule.inject_T(injection_patch)
            prev_injection_flag = not prev_injection_flag
            schedule.merge([(0,0), injection_patch], [])
            schedule.discard([injection_patch])
            schedule.conditional_S((0,0), len(schedule.all_instructions) - 2)
            
        schedule.discard([(0,0)])

        self.schedule = schedule

class RandomTSchedule:
    """Injects a specified number of T gates on one patch, with a random number
    of idle rounds between each T gate. Alternates T state creation between two
    adjacent patches.
    """
    def __init__(self, num_Ts: int, max_idle_between_Ts: int):
        """Builds the schedule.
        
        Args:
            num_Ts: Number of T gates to inject.
            max_idle_between_Ts: Maximum number of idle rounds between each T gate.
        """
        schedule = LatticeSurgerySchedule()
        prev_injection_flag = False
        for i in range(num_Ts):
            schedule.idle([(0,0)], np.random.randint(1, max_idle_between_Ts))

            injection_patch = (0,1) if prev_injection_flag else (1,0)
            schedule.inject_T(injection_patch)
            prev_injection_flag = not prev_injection_flag
            schedule.merge([(0,0), injection_patch], [])
            schedule.discard([injection_patch])
            schedule.conditional_S((0,0), len(schedule.all_instructions) - 2)

        schedule.discard([(0,0)])

        self.schedule = schedule