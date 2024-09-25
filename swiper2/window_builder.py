from dataclasses import dataclass
import math

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
    parent_instr: Instruction


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

        new_windows = []
        instr_patch_groups = {}
        for round in self._waiting_rounds:
            instr, patch = round.instruction, round.patch
            instr_patch_groups.setdefault((instr, patch), []).append(round)
            
        for (instr, patch), rounds in instr_patch_groups.items():
            min_round = min(rounds, key=lambda x: x.round)
            max_round = max(rounds, key=lambda x: x.round)
            # TODO: how to handle arbitrary duration idles
            expected_duration = min(math.ceil(0.5 * self.d * (instr.duration if isinstance(instr.duration, int) else instr.duration.value)),
                                    self.d)
        
            if (max_round.round - min_round.round) + 1 < expected_duration:
                # Not enough rounds to create a window
                continue
            commit_region = SpacetimeRegion(space_footprint=[patch],
                                            round_start=min_round.round,
                                            duration=expected_duration)
            new_windows.append(DecodingWindow(blocking=instr.conditioned_on_idx is not None, 
                                              commit_region=commit_region, parent_instr=instr,
                                              buffer_regions=[], decoding_time=0))
            for round in rounds:
                self._waiting_rounds.remove(round)

        return new_windows
