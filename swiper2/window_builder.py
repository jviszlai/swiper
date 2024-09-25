from dataclasses import dataclass

from swiper2.lattice_surgery_schedule import Instruction
from swiper2.device_manager import SyndromeRound

@dataclass
class SpacetimeRegion:
    '''
    A region of spacetime in the decoding volume

    Attributes:
        space_footprint: spatial coordinates of patches in region
        round_start: measurement round starting the region
        duration: temporal length in units of measurement rounds
    '''
    space_footprint: list[tuple[int, int]]
    round_start: int
    duration: int

    def contains_syndrome_round(self, syndrome_round: SyndromeRound) -> bool:
        '''
        Check if a syndrome round is contained in this region
        '''
        return syndrome_round.patch in self.space_footprint and self.round_start <= syndrome_round.round < self.round_start + self.duration

@dataclass
class DecodingWindow:
    '''
    Attributes:
        blocking: Whether the commit region of this window contains 
                  a blocking operation.
        commit_region: Spacetime region that is commited after decoding.
        buffer_regions: Spacetime regions that are not commited after decoding.
                        The boundary between a buffer region and the commit region
                        forms a decoding dependency from this window to
                        adjacent windows.
        decoding_time: Number of rounds required to decode this window.
        parent_instrs: Instructions that are at least partially contained in the 
                       commit region of this window.
        
    '''
    blocking: bool
    commit_region: SpacetimeRegion
    buffer_regions: list[SpacetimeRegion]
    decoding_time: int
    parent_instrs: list[Instruction]


class WindowBuilder():

    def __init__(self, d: int) -> None:
        self._waiting_rounds = []
        self.d = d

    def build_windows(self, 
                      new_rounds: list[SyndromeRound]
                      ) -> list[DecodingWindow]:
        '''
        TODO
        '''
        if not new_rounds or len(new_rounds) == 0:
            return []
        
        self._waiting_rounds.extend(new_rounds)

        min_round = min(self._waiting_rounds, key=lambda x: x.round)
        max_round = max(self._waiting_rounds, key=lambda x: x.round)
        if max_round.round - min_round.round < self.d:
            # Not enough rounds to create a window
            return []
        new_windows = []
        active_patches = set([round.patch for round in self._waiting_rounds])
        patch_instrs = {patch: [] for patch in active_patches}
        for round in self._waiting_rounds:
            if round.instruction not in patch_instrs[round.patch]:
                patch_instrs[round.patch].append(round.instruction)
        for patch, instrs in patch_instrs.items():
            commit_region = SpacetimeRegion(space_footprint=[patch],
                                            round_start=min_round.round,
                                            duration=self.d)
            blocking = any([instr.conditioned_on_idx is not None for instr in instrs])
            new_windows.append(DecodingWindow(blocking=blocking, commit_region=commit_region, parent_instrs=instrs,
                                              buffer_regions=[], decoding_time=0))

        return new_windows
