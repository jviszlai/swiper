from dataclasses import dataclass
import math
import numpy as np

from swiper.lattice_surgery_schedule import Instruction
from swiper.device_manager import SyndromeRound

@dataclass(frozen=True)
class SpacetimeRegion:
    """A region of spacetime in the decoding volume.

    Attributes:
        patch: Spatial coordinates of region.
        round_start: Measurement round starting the region.
        duration: Temporal length, in units of measurement rounds.
        num_spatial_boundaries: Number of spatial faces which are not shared
            with another region.
        initialized_patch: If True, this region was the first of the patch.
        discard_after: If True, this patch was discarded after this region.
        prior_t: If True, a T gate was injected in the round before this region.
        merge_instr: MERGE instruction if applicable
    """
    patch: tuple[int, int]
    round_start: int
    duration: int
    num_spatial_boundaries: int
    initialized_patch: bool = False
    discard_after: bool = False
    prior_t: bool = False
    merge_instr: Instruction | None = None

    def contains_syndrome_round(self, *, patch: tuple[int, int] | None = None, round: int | None = None, syndrome_round: SyndromeRound | None = None) -> bool:
        """Check if a syndrome round is contained in this region. Either patch
        and round must be given, or a syndrome round object.
        """
        if patch is not None and round is not None:
            return patch == self.patch and self.round_start <= round < self.round_start + self.duration
        elif syndrome_round is not None:
            return syndrome_round.patch == self.patch and self.round_start <= syndrome_round.round < self.round_start + self.duration
        else:
            raise ValueError('Either patch and round or a syndrome round must be given.')
    
    def shares_timelike_boundary(self, other: 'SpacetimeRegion') -> bool:
        """Check if this region shares a timelike boundary with another region.
        """
        return (
            self.patch == other.patch
            and
            (
                (
                    (not other.discard_after)
                    and
                    self.round_start == other.round_start + other.duration
                )
                or 
                (
                    (not self.discard_after)
                    and
                    other.round_start == self.round_start + self.duration
                )
            )
        )
    
    def shares_spacelike_boundary(self, other: 'SpacetimeRegion') -> bool:
        """Check if this region shares a spacelike boundary with another region.
        """
        return (
            self.merge_instr is not None and other.merge_instr is not None
            and self.merge_instr == other.merge_instr
            and ((self.patch, other.patch) in self.merge_instr.merge_faces or (other.patch, self.patch) in self.merge_instr.merge_faces)
        )
    
    def shares_boundary(self, other: 'SpacetimeRegion') -> bool:
        return self.shares_timelike_boundary(other) or self.shares_spacelike_boundary(other)
    
    def overlaps(self, other: 'SpacetimeRegion') -> bool:
        """Check if this region overlaps with another region.
        """
        return self.patch == other.patch and self.round_start < other.round_start + other.duration and other.round_start < self.round_start + self.duration
    
    def __repr__(self):
        return f'Region({self.patch}, {self.round_start}, {self.duration}, {self.num_spatial_boundaries}, {self.initialized_patch}, {self.discard_after})'

@dataclass(frozen=True)
class DecodingWindow:
    """A decoding window with commit and buffer regions.

    Attributes:
        commit_region: Spacetime region that is commited after decoding.
        buffer_regions: Spacetime regions that are not commited after decoding.
                        The boundary between a buffer region and the commit region
                        forms a decoding dependency from this window to
                        adjacent windows.
        merge_instr: MERGE instruction for spatial buffers if necessary.
        parent_instr_idx: List of indices of instructions that generated this window.
        constructed: True if window is finished being constructed with buffers 
    """
    commit_region: tuple[SpacetimeRegion, ...]
    buffer_regions: frozenset[SpacetimeRegion]
    merge_instr: frozenset[Instruction]
    parent_instr_idx: frozenset[int]
    window_idx: int
    constructed: bool

    def total_spacetime_volume(self) -> int:
        """Calculate the total spacetime volume of this window, in units of
        rounds*d^2."""
        if isinstance(self.commit_region, SpacetimeRegion):
            return self.commit_region.duration + sum(region.duration for region in self.buffer_regions)
        else:
            return sum(region.duration for region in self.commit_region) + sum(region.duration for region in self.buffer_regions)

    def shares_timelike_boundary(self, other: 'DecodingWindow') -> bool:
        for region in self.commit_region:
            for other_region in other.commit_region:
                if region.shares_timelike_boundary(other_region):
                    return True
        return False

    def shared_spacelike_boundaries(self, other: 'DecodingWindow') -> list[tuple[SpacetimeRegion, SpacetimeRegion]]:
        shared_boundaries = []
        for region in self.commit_region:
            for other_region in other.commit_region:
                if region.shares_spacelike_boundary(other_region):
                    shared_boundaries.append((region, other_region))
        return shared_boundaries
    
    def shares_spacelike_boundary(self, other: 'DecodingWindow') -> bool:
        for region in self.commit_region:
            for other_region in other.commit_region:
                if region.shares_spacelike_boundary(other_region):
                    return True
        return False

    def shares_boundary(self, other: 'DecodingWindow') -> bool:
        return self.shares_timelike_boundary(other) or self.shares_spacelike_boundary(other)

    def get_touching_commit_regions(self, other: 'DecodingWindow') -> list[SpacetimeRegion]:
        """Get the commit regions of `other` that are touching (share a
        boundary)."""
        shared_boundaries = self.shared_spacelike_boundaries(other)
        adjacent_regions = []
        for region in self.commit_region:
            for other_region in other.commit_region:
                if (region, other_region) in shared_boundaries or (other_region, region) in shared_boundaries:
                    adjacent_regions.append(other_region)
        return adjacent_regions

    def overlaps(self, other: 'DecodingWindow') -> bool:
        """Check if buffer regions of this window overlap with commit regions of
        other, or vice versa."""
        for self_commit in self.commit_region:
            for other_buffer in other.buffer_regions:
                if self_commit.overlaps(other_buffer):
                    return True
        for other_commit in other.commit_region:
            for self_buffer in self.buffer_regions:
                if other_commit.overlaps(self_buffer):
                    return True
        return False

    def buffer_boundary_commits(self) -> dict[SpacetimeRegion, list[SpacetimeRegion]]:
        """Returns dict mapping each buffer region to all the commit regions
        touching it."""
        commits = {}
        for br in self.buffer_regions:
            for cr in self.commit_region:
                if br.shares_boundary(cr):
                    commits.setdefault(br, []).append(cr)
        return commits
    
    def __repr__(self):
        return f'Window({self.commit_region}, {self.buffer_regions}, {self.parent_instr_idx}, {self.constructed})'

class WindowBuilder():
    def __init__(self, d: int, lightweight_output: bool = False) -> None:
        self._patch_groups: dict[tuple[int, int], list[int]] = {}
        self._all_rounds: list[SyndromeRound] = []
        self._waiting_rounds: set[int] = set()
        self._inject_t_rounds: set[int] = set()
        self._inject_t_rounds_dict: dict[tuple[int, int], list[int]] = dict()
        self._total_rounds_processed: int = 0
        self._created_window_count: int = 0
        self.d: int = d
        self.lightweight_output = lightweight_output

    def build_windows(
            self, 
            new_rounds: list[SyndromeRound],
        ) -> list[DecodingWindow]:
        """Process new rounds and output any windows with complete commit
        regions.
        
        Args:
            new_rounds: List of new syndrome rounds to process. Should all be
                from the same cycle of the device.

        Returns:
            List of newly-completed decoding windows.
        """
        if not new_rounds or len(new_rounds) == 0:
            # Time to chug through that backlog
            curr_round = -1
        else:
            curr_round = new_rounds[0].round
            assert all([round.round == curr_round for round in new_rounds])

            new_round_start = len(self._all_rounds)
            self._all_rounds.extend(new_rounds)
            self._waiting_rounds.update([i+new_round_start
                                        for i,round in enumerate(new_rounds)
                                        if round.instruction.name != 'INJECT_T']) # T injection is not decoded 
            for i,round in enumerate(new_rounds):
                if round.instruction.name != 'INJECT_T':
                    self._patch_groups.setdefault(round.patch, []).append(i+new_round_start)
                else:
                    self._inject_t_rounds_dict.setdefault(round.patch, []).append(i+new_round_start)

        new_windows = []

        for patch, round_indices in list(self._patch_groups.items()):
            rounds = [self._all_rounds[round_idx] for round_idx in round_indices]
            min_round = min(rounds, key=lambda x: x.round)
            max_round = max(rounds, key=lambda x: x.round)
            duration = self.d
            if max_round.round != curr_round or max_round.discard_after:
                # Dangling rounds (e.g. S gate cap)
                duration = max_round.round - min_round.round + 1
            elif min_round.instruction.name != 'MERGE' and max_round.instruction.name == 'MERGE':
                # Aligning windows with merges is non-negotiable due to the need for spatial buffers
                junk_round_end = max(rounds, key=lambda x: x.round * (0 if x.instruction.name == 'MERGE' else 1))
                round_indices = [round_idx for round_idx in round_indices if self._all_rounds[round_idx].round <= junk_round_end.round]
                rounds = [self._all_rounds[round_idx] for round_idx in round_indices]
                duration = junk_round_end.round - min_round.round + 1
            elif min_round.instruction.name == 'MERGE' and max_round.instruction.name != 'MERGE':
                junk_round_end = max(rounds, key=lambda x: x.round * (1 if x.instruction.name == 'MERGE' else 0))
                round_indices = [round_idx for round_idx in round_indices if self._all_rounds[round_idx].round <= junk_round_end.round]
                rounds = [self._all_rounds[round_idx] for round_idx in round_indices]
                duration = junk_round_end.round - min_round.round + 1
            elif (max_round.round - min_round.round) + 1 < duration:
                # Not enough rounds to create a window
                continue
            max_round = max(rounds, key=lambda x: x.round)
            num_spatial_boundaries = 4 - sum(patch in face for face in min_round.instruction.merge_faces)
            prior_t = False
            if not min_round.initialized_patch and patch in self._inject_t_rounds_dict:
                for idx in reversed(self._inject_t_rounds_dict[patch]):
                    t_round = self._all_rounds[idx]
                    if t_round.round == min_round.round-1:
                        prior_t = True
                        break
                    elif t_round.round < min_round.round-1:
                        break
            parent_instr_idx = frozenset([round.instruction_idx for round in rounds])
            commit_region = SpacetimeRegion(
                patch=patch,
                round_start=min_round.round,
                duration=duration,
                num_spatial_boundaries=num_spatial_boundaries,
                initialized_patch=min_round.initialized_patch,
                discard_after=max_round.discard_after,
                prior_t=prior_t,
                merge_instr=min_round.instruction if min_round.instruction.name == 'MERGE' else None,
            )
            new_windows.append(DecodingWindow(
                commit_region=(commit_region,),
                buffer_regions=frozenset(),
                merge_instr=frozenset() if min_round.instruction.name != 'MERGE' else frozenset([min_round.instruction]),
                parent_instr_idx=parent_instr_idx,
                window_idx=self._created_window_count,
                constructed=False,
            ))
            self._created_window_count += 1

            self._patch_groups[patch] = [round_idx for round_idx in self._patch_groups[patch] if round_idx not in round_indices]
            if not self._patch_groups[patch]:
                self._patch_groups.pop(patch)
            self._waiting_rounds -= set(round_indices)
            if self.lightweight_output:
                for round_idx in round_indices:
                    self._all_rounds[round_idx] = None

        self._total_rounds_processed += len(new_rounds)

        return new_windows

    def flush(self):
        """Flush all remaining rounds into windows, allowing smaller windows
        than the usual size.
        """
        new_windows = []
        for patch, round_indices in list(self._patch_groups.items()):
            rounds = [self._all_rounds[round_idx] for round_idx in round_indices]
            min_round = min(rounds, key=lambda x: x.round)
            max_round = max(rounds, key=lambda x: x.round)
            duration = max_round.round - min_round.round + 1
            num_spatial_boundaries = 4 - sum(patch in face for face in min_round.instruction.merge_faces)
            parent_instr_idx = frozenset([round.instruction_idx for round in rounds])
            commit_region = SpacetimeRegion(
                patch=patch,
                round_start=min_round.round,
                duration=duration,
                num_spatial_boundaries=num_spatial_boundaries,
                initialized_patch=min_round.initialized_patch,
                discard_after=True,
                merge_instr=min_round.instruction if min_round.instruction.name == 'MERGE' else None,
            )
            new_windows.append(DecodingWindow(
                commit_region=(commit_region,),
                buffer_regions=frozenset(),
                merge_instr=frozenset() if min_round.instruction.name != 'MERGE' else frozenset([min_round.instruction]),
                parent_instr_idx=parent_instr_idx,
                window_idx=self._created_window_count,
                constructed=False,
            ))
            self._created_window_count += 1
            self._patch_groups.pop(patch)
            self._waiting_rounds -= set(round_indices)

        assert len(self._waiting_rounds) == 0
        return new_windows
    
    def get_incomplete_instructions(self):
        return {self._all_rounds[round].instruction_idx for round in self._waiting_rounds}