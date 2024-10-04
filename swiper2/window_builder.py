from dataclasses import dataclass
import math

from swiper2.lattice_surgery_schedule import Instruction
from swiper2.device_manager import SyndromeRound

@dataclass(frozen=True)
class SpacetimeRegion:
    '''
    A region of spacetime in the decoding volume

    Attributes:
        patch: spatial coordinates of region
        round_start: measurement round starting the region
        duration: temporal length in units of measurement rounds
    '''
    patch: tuple[int, int]
    round_start: int
    duration: int

    def contains_syndrome_round(self, syndrome_round: SyndromeRound) -> bool:
        '''
        Check if a syndrome round is contained in this region
        '''
        return syndrome_round.patch == self.patch and self.round_start <= syndrome_round.round < self.round_start + self.duration

@dataclass(frozen=True)
class DecodingWindow:
    '''
    Attributes:
        commit_region: Spacetime region that is commited after decoding.
        buffer_regions: Spacetime regions that are not commited after decoding.
                        The boundary between a buffer region and the commit region
                        forms a decoding dependency from this window to
                        adjacent windows.
        merge_instr: MERGE instruction for spatial buffers if necessary.
        parent_instr_idx: List of indices of instructions that generated this window.
        constructed: True if window is finished being constructed with buffers
        
    '''
    commit_region: SpacetimeRegion
    buffer_regions: frozenset[SpacetimeRegion]
    merge_instr: Instruction | None
    parent_instr_idx: frozenset[int]
    constructed: bool

    def total_spacetime_volume(self) -> int:
        '''
        Calculate the total spacetime volume of this window, in units of
        rounds*d^2.
        '''
        return self.commit_region.duration + sum([region.duration for region in self.buffer_regions])


class WindowBuilder():

    def __init__(self, d: int, enforce_alignment: bool) -> None:
        self._waiting_rounds = []
        self.d = d
        self.enforce_alignment = enforce_alignment

    def build_windows(self, 
                      new_rounds: list[SyndromeRound]
                      ) -> list[DecodingWindow]:
        '''
        TODO
        '''
        if not new_rounds or len(new_rounds) == 0:
            # Time to chug through that backlog
            curr_round = -1
        else:
            curr_round = new_rounds[0].round
            assert all([round.round == curr_round for round in new_rounds])
        
            self._waiting_rounds.extend([round 
                                        for round in new_rounds
                                        if round.instruction.name != 'INJECT_T']) # T injection is not decoded 
        new_windows = []

        if not self.enforce_alignment:
            patch_groups = {}
            for round in self._waiting_rounds:
                patch_groups.setdefault(round.patch, []).append(round)
            
            for patch, rounds in patch_groups.items():
                min_round = min(rounds, key=lambda x: x.round)
                max_round = max(rounds, key=lambda x: x.round)
                duration = self.d
                if max_round.round != curr_round:
                    # Dangling rounds (e.g. S gate cap)
                    duration = max_round.round - min_round.round + 1
                elif (max_round.round - min_round.round) + 1 < duration:
                    # Not enough rounds to create a window
                    continue
                elif min_round.instruction.name != 'MERGE' and max_round.instruction.name == 'MERGE':
                    # Aligning windows with merges is non-negotiable due to the need for spatial buffers
                    junk_round_end = max(rounds, key=lambda x: x.round * (0 if x.instruction.name == 'MERGE' else 1))
                    rounds = [round for round in rounds if round.round <= junk_round_end.round]
                    duration = junk_round_end.round - min_round.round + 1
                parent_instr_idx = frozenset([round.instruction_idx for round in rounds])
                commit_region = SpacetimeRegion(patch=patch,
                                                round_start=min_round.round,
                                                duration=duration)
                new_windows.append(DecodingWindow(commit_region=commit_region,
                                                  buffer_regions=frozenset(),
                                                  merge_instr=None if min_round.instruction.name != 'MERGE' else min_round.instruction,
                                                  parent_instr_idx=parent_instr_idx,
                                                  constructed=False))
                for round in rounds:
                    self._waiting_rounds.remove(round)
                    
        else:
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
                new_windows.append(DecodingWindow(commit_region=commit_region, 
                                                  buffer_regions=[], decoding_time=0))
                for round in rounds:
                    self._waiting_rounds.remove(round)

        return new_windows
