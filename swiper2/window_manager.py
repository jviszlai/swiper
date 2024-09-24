from dataclasses import dataclass
import networkx as nx

from swiper2.lattice_surgery_schedule import Instruction

@dataclass
class SpacetimeRegion:
    '''
    A region of spacetime in the decoding volume

    Attributes:
        space_footprint: spatial coordinates of patches in region
        duration: temporal length in units of measurement rounds
    '''
    space_footprint: list[tuple[int, int]]
    duration: int


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

class WindowManager:
    def __init__(self):
        self.waiting_windows = []
        self.window_dag = nx.DiGraph()
    

