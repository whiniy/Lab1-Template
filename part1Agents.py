from model import (
    Location,
    Portal,
    EmptyEntity,
    Wizard,
    Goblin,
    Crystal,
    WizardMoves,
    GoblinMoves,
    GameAction,
    GameState,
)
from agents import WizardSearchAgent
import heapq
from dataclasses import dataclass


class WizardDFS(WizardSearchAgent):
    @dataclass(eq=True, frozen=True, order=True)
    class SearchState:
        wizard_loc: Location
        portal_loc: Location

    paths: dict[SearchState, list[WizardMoves]] = {}
    search_stack: list[SearchState] = []
    initial_game_state: GameState

    def search_to_game(self, search_state: SearchState) -> GameState:
        initial_wizard_loc = self.initial_game_state.active_entity_location
        initial_wizard = self.initial_game_state.get_active_entity()

        new_game_state = (
            self.initial_game_state.replace_entity(
                initial_wizard_loc.row, initial_wizard_loc.col, EmptyEntity()
            )
            .replace_entity(
                search_state.wizard_loc.row, search_state.wizard_loc.col, initial_wizard
            )
            .replace_active_entity_location(search_state.wizard_loc)
        )

        return new_game_state

    def game_to_search(self, game_state: GameState) -> SearchState:
        wizard_loc = game_state.active_entity_location
        portal_loc = game_state.get_all_tile_locations(Portal)[0]
        return self.SearchState(wizard_loc, portal_loc)

    def __init__(self, initial_state: GameState):
        self.start_search(initial_state)

    def start_search(self, game_state: GameState):
        self.initial_game_state = game_state

        initial_search_state = self.game_to_search(game_state)
        self.paths = {}
        self.paths[initial_search_state] = []
        self.search_stack = [initial_search_state]

    def is_goal(self, state: SearchState) -> bool:
        return state.wizard_loc == state.portal_loc

    def next_search_expansion(self) -> GameState | None:
        # check if stack is empty before popping to avoid error
        if not self.search_stack:
            return None
        
        # pop off last element in stack, then convert it to a game state
        return self.search_to_game(self.search_stack.pop())

    def process_search_expansion(
        self, source: GameState, target: GameState, action: WizardMoves
    ) -> None:
        sourceState = self.game_to_search(source)
        targetState = self.game_to_search(target)

        # check if targetState is already visited/in paths
        if targetState in self.paths:
            return

        # adds new move that gets us from sourceState to targetState
        self.paths[targetState] = self.paths[sourceState] + [action]

        # check if targetState is the goal, 
        if self.is_goal(targetState):
            # set path to targetState as the plan
            # reverse the path since we want to pop moves off
            self.plan = self.paths[targetState][::-1]
        else:
            # add targetState to the stack to be explored later
            self.search_stack.append(targetState)


class WizardBFS(WizardSearchAgent):
    @dataclass(eq=True, frozen=True, order=True)
    class SearchState:
        wizard_loc: Location
        portal_loc: Location

    paths: dict[SearchState, list[WizardMoves]] = {}
    search_stack: list[SearchState] = []
    initial_game_state: GameState

    def search_to_game(self, search_state: SearchState) -> GameState:
        initial_wizard_loc = self.initial_game_state.active_entity_location
        initial_wizard = self.initial_game_state.get_active_entity()

        new_game_state = (
            self.initial_game_state.replace_entity(
                initial_wizard_loc.row, initial_wizard_loc.col, EmptyEntity()
            )
            .replace_entity(
                search_state.wizard_loc.row, search_state.wizard_loc.col, initial_wizard
            )
            .replace_active_entity_location(search_state.wizard_loc)
        )

        return new_game_state

    def game_to_search(self, game_state: GameState) -> SearchState:
        wizard_loc = game_state.active_entity_location
        portal_loc = game_state.get_all_tile_locations(Portal)[0]
        return self.SearchState(wizard_loc, portal_loc)

    def __init__(self, initial_state: GameState):
        self.start_search(initial_state)

    def start_search(self, game_state: GameState):
        self.initial_game_state = game_state

        initial_search_state = self.game_to_search(game_state)
        self.paths = {}
        self.paths[initial_search_state] = []
        self.search_stack = [initial_search_state]

    def is_goal(self, state: SearchState) -> bool:
        return state.wizard_loc == state.portal_loc

    def next_search_expansion(self) -> GameState | None:
        # check if state is empty before popping to avoid error
        if not self.search_stack:
            return None
        
        # pop off first element in stack, then convert it to a game state
        return self.search_to_game(self.search_stack.pop(0))

    def process_search_expansion(
        self, source: GameState, target: GameState, action: WizardMoves
    ) -> None:
        # convert source and target game states to search states
        sourceState = self.game_to_search(source)
        targetState = self.game_to_search(target)

        # check if targetState is already visited/in paths
        if targetState in self.paths:
            return

        # adds new move that gets us from sourceState to targetState
        self.paths[targetState] = self.paths[sourceState] + [action]

        # check if targetState is the goal
        if self.is_goal(targetState):
            # set path to targetState as the plan
            # reverse the path since we want to pop moves off
            self.plan = self.paths[targetState][::-1]
        else:

            # add targetState to the stack to be explored later
            self.search_stack.append(targetState)


class WizardAstar(WizardSearchAgent):
    @dataclass(eq=True, frozen=True, order=True)
    class SearchState:
        wizard_loc: Location
        portal_loc: Location

    paths: dict[SearchState, tuple[float, list[WizardMoves]]] = {}
    search_pq: list[tuple[float, SearchState]] = []
    initial_game_state: GameState

    def search_to_game(self, search_state: SearchState) -> GameState:
        initial_wizard_loc = self.initial_game_state.active_entity_location
        initial_wizard = self.initial_game_state.get_active_entity()

        new_game_state = (
            self.initial_game_state.replace_entity(
                initial_wizard_loc.row, initial_wizard_loc.col, EmptyEntity()
            )
            .replace_entity(
                search_state.wizard_loc.row, search_state.wizard_loc.col, initial_wizard
            )
            .replace_active_entity_location(search_state.wizard_loc)
        )

        return new_game_state

    def game_to_search(self, game_state: GameState) -> SearchState:
        wizard_loc = game_state.active_entity_location
        portal_loc = game_state.get_all_tile_locations(Portal)[0]
        return self.SearchState(wizard_loc, portal_loc)

    def __init__(self, initial_state: GameState):
        self.start_search(initial_state)

    def start_search(self, game_state: GameState):
        self.initial_game_state = game_state

        initial_search_state = self.game_to_search(game_state)
        self.paths = {}
        self.paths[initial_search_state] = 0, []
        self.search_pq = [(0, initial_search_state)]

    def is_goal(self, state: SearchState) -> bool:
        return state.wizard_loc == state.portal_loc

    def cost(self, source: GameState, target: GameState, action: WizardMoves) -> float:
        return 1

    def heuristic(self, target: GameState) -> float:
        targetState = self.game_to_search(target)

        # Manhattan distance heuristic
        return abs(targetState.wizard_loc.row - targetState.portal_loc.row) + abs(targetState.wizard_loc.col - targetState.portal_loc.col)

    def next_search_expansion(self) -> GameState | None:
        # check if priority queue is empty before popping to avoid error
        if not self.search_pq:
            return None
        
        # pop off element with lowest f(n) value, then convert it to a game state
        return self.search_to_game(self.search_pq.pop(0)[1])

    def process_search_expansion(
        self, source: GameState, target: GameState, action: WizardMoves
    ) -> None:
        sourceState = self.game_to_search(source)
        targetState = self.game_to_search(target)

        # source cost and path
        sourceCost, sourcePath = self.paths[sourceState]
        # target cost
        targetCost = sourceCost + self.cost(source, target, action)

        if targetState in self.paths:
            # compare new target cost with old target cost
            oldTargetCost, _ = self.paths[targetState]
            # ignore if target cost is higher than old target cost
            if targetCost >= oldTargetCost:
                return
            
        # update cost and path
        # first time seeing this state or found a cheaper path to this state
        self.paths[targetState] = (targetCost, sourcePath + [action])

        # check if targetState is the goal
        if self.is_goal(targetState):
            # set path to targetState as the plan
            # reverse the path since we want to pop moves off
            self.plan = self.paths[targetState][1][::-1]
        else:
            f = targetCost + self.heuristic(target)
            # add targetState to the priority queue to be explored later
            self.search_pq.append((f, targetState))
            self.search_pq.sort(key=lambda x: x[0])        

class CrystalSearchWizard(WizardSearchAgent):
    @dataclass(eq=True, frozen=True, order=True)

    # locations and which crystals are left to be collected
    class SearchState:
        wizard_loc: Location
        portal_loc: Location
        crystals_left: frozenset[Location]

    paths: dict[SearchState, tuple[float, list[WizardMoves]]] = {}
    search_pq: list[tuple[float, SearchState]] = []
    initial_game_state: GameState

    def search_to_game(self, search_state: SearchState) -> GameState:
        initial_wizard_loc = self.initial_game_state.active_entity_location
        initial_wizard = self.initial_game_state.get_active_entity()

        new_game_state = self.initial_game_state.replace_entity(
            initial_wizard_loc.row, initial_wizard_loc.col, EmptyEntity()
        )

        # remove crystals from the initial map
        for crystal_loc in self.initial_game_state.get_all_entity_locations(Crystal):
            new_game_state = new_game_state.replace_entity(
                crystal_loc.row, crystal_loc.col, EmptyEntity()
            )

        # add back crystals that are left
        for crystal_loc in search_state.crystals_left:
            new_game_state = new_game_state.replace_entity(
                crystal_loc.row, crystal_loc.col, Crystal()
            )

        # place the wizard in search location
        new_game_state = new_game_state.replace_entity(
            search_state.wizard_loc.row, search_state.wizard_loc.col, initial_wizard
        ).replace_active_entity_location(search_state.wizard_loc)

        return new_game_state

    def game_to_search(self, game_state: GameState) -> SearchState:
        wizard_loc = game_state.active_entity_location
        portal_loc = game_state.get_all_tile_locations(Portal)[0]
        crystals_left = frozenset(game_state.get_all_entity_locations(Crystal))
        return self.SearchState(wizard_loc, portal_loc, crystals_left)


    def __init__(self, initial_state: GameState):
        self.start_search(initial_state)

    def start_search(self, game_state: GameState):
        self.initial_game_state = game_state

        initial_search_state = self.game_to_search(game_state)
        self.paths = {}
        self.paths[initial_search_state] = 0, []
        self.search_pq = [(0, initial_search_state)]


    def is_goal(self, state: SearchState) -> bool:
        return state.wizard_loc == state.portal_loc and not state.crystals_left

    def cost(self, source: GameState, target: GameState, action: WizardMoves) -> float:
        return 1

    def heuristic(self, target: GameState) -> float:
        targetState = self.game_to_search(target)

        # Manhattan distance
        def manhat(loc1: Location, loc2: Location) -> float:
            return abs(loc1.row - loc2.row) + abs(loc1.col - loc2.col)

        if not targetState.crystals_left:
            # if no crystals left, heuristic is just distance to portal
            return manhat(targetState.wizard_loc, targetState.portal_loc)
        
        closest_crystal_dist = min(manhat(targetState.wizard_loc, crystal_loc) for crystal_loc in targetState.crystals_left)
        farthest_crystal_dist = max(manhat(crystal_loc, targetState.portal_loc) for crystal_loc in targetState.crystals_left)

        # take into account distance to closest crystal and distance from farthest crystal to portal
        return closest_crystal_dist + farthest_crystal_dist

    def next_search_expansion(self) -> GameState | None:
        # check if priority queue is empty before popping to avoid error
        if not self.search_pq:
            return None
        
        # pop off element with lowest f(n) value, then convert it to a game state
        return self.search_to_game(heapq.heappop(self.search_pq)[1])

    def process_search_expansion(
        self, source: GameState, target: GameState, action: WizardMoves
    ) -> None:
        sourceState = self.game_to_search(source)
        targetState = self.game_to_search(target)

        # source cost and path
        sourceCost, sourcePath = self.paths[sourceState]
        # target cost
        targetCost = sourceCost + self.cost(source, target, action)

        if targetState in self.paths:
            # compare new target cost with old target cost
            oldTargetCost, _ = self.paths[targetState]
            # ignore if target cost is higher than old target cost
            if targetCost >= oldTargetCost:
                return
            
        # update cost and path
        # first time seeing this state or found a cheaper path to this state
        self.paths[targetState] = (targetCost, sourcePath + [action])

        if self.is_goal(targetState):
            # set path to targetState as the plan
            # reverse the path since we want to pop moves off
            self.plan = self.paths[targetState][1][::-1]
        else:
            f = targetCost + self.heuristic(target)
            # add targetState to the priority queue to be explored later
            heapq.heappush(self.search_pq, (f, targetState))

class SuboptimalCrystalSearchWizard(CrystalSearchWizard):
    @dataclass(eq=True, frozen=True, order=True)

    class SearchState:
        wizard_loc: Location
        portal_loc: Location
        crystals_left: frozenset[Location]
    
    paths: dict[SearchState, tuple[float, list[WizardMoves]]] = {}
    search_pq: list[tuple[float, SearchState]] = []
    initial_game_state: GameState

    def search_to_game(self, search_state: SearchState) -> GameState:
        initial_wizard_loc = self.initial_game_state.active_entity_location
        initial_wizard = self.initial_game_state.get_active_entity()

        new_game_state = self.initial_game_state.replace_entity(
            initial_wizard_loc.row, initial_wizard_loc.col, EmptyEntity()
        )

        # remove crystals from the initial map
        for crystal_loc in self.initial_game_state.get_all_entity_locations(Crystal):
            new_game_state = new_game_state.replace_entity(
                crystal_loc.row, crystal_loc.col, EmptyEntity()
            )

        # add back crystals that are left
        for crystal_loc in search_state.crystals_left:
            new_game_state = new_game_state.replace_entity(
                crystal_loc.row, crystal_loc.col, Crystal()
            )

        # place the wizard in search location
        new_game_state = new_game_state.replace_entity(
            search_state.wizard_loc.row, search_state.wizard_loc.col, initial_wizard
        ).replace_active_entity_location(search_state.wizard_loc)

        return new_game_state
    
    def game_to_search(self, game_state: GameState) -> SearchState:
        wizard_loc = game_state.active_entity_location
        portal_loc = game_state.get_all_tile_locations(Portal)[0]
        crystals_left = frozenset(game_state.get_all_entity_locations(Crystal))
        return self.SearchState(wizard_loc, portal_loc, crystals_left)


    def __init__(self, initial_state: GameState):
        self.start_search(initial_state)

    def start_search(self, game_state: GameState):
        self.initial_game_state = game_state

        initial_search_state = self.game_to_search(game_state)
        self.paths = {}
        self.paths[initial_search_state] = 0, []
        self.search_pq = [(0, initial_search_state)]

    def is_goal(self, state: SearchState) -> bool:
        return state.wizard_loc == state.portal_loc and not state.crystals_left

    def cost(self, source: GameState, target: GameState, action: WizardMoves) -> float:
        return 1

    def heuristic(self, target: SearchState) -> float:
        # Manhattan distance
        def manhat(loc1: Location, loc2: Location) -> float:
            return abs(loc1.row - loc2.row) + abs(loc1.col - loc2.col)

        if not target.crystals_left:
            # if no crystals left, heuristic is just distance to portal
            return manhat(target.wizard_loc, target.portal_loc)
        
        closest_crystal_dist = min(manhat(target.wizard_loc, crystal_loc) for crystal_loc in target.crystals_left)
        farthest_crystal_dist = max(manhat(crystal_loc, target.portal_loc) for crystal_loc in target.crystals_left)

        # take into account distance to closest crystal and distance from farthest crystal to portal
        return 1.5 * (closest_crystal_dist + farthest_crystal_dist)
    
    def next_search_expansion(self) -> GameState | None:
        # check if priority queue is empty before popping to avoid error
        if not self.search_pq:
            return None
        
        # pop off element with lowest f(n) value, then convert it to a game state
        return self.search_to_game(self.search_pq.pop(0)[1])

    def process_search_expansion(
        self, source: GameState, target: GameState, action: WizardMoves
    ) -> None:
        sourceState = self.game_to_search(source)
        targetState = self.game_to_search(target)

        # source cost and path
        sourceCost, sourcePath = self.paths[sourceState]
        # target cost
        targetCost = sourceCost + self.cost(source, target, action)

        if targetState in self.paths:
            # compare new target cost with old target cost
            oldTargetCost, _ = self.paths[targetState]
            # ignore if target cost is higher than old target cost
            if targetCost >= oldTargetCost:
                return
            
        # update cost and path
        # first time seeing this state or found a cheaper path to this state
        self.paths[targetState] = (targetCost, sourcePath + [action])

        if self.is_goal(targetState):
            # set path to targetState as the plan
            # reverse the path since we want to pop moves off
            self.plan = self.paths[targetState][1][::-1]
        else:
            f = targetCost + self.heuristic(self.game_to_search(target))
            # add targetState to the priority queue to be explored later
            self.search_pq.append((f, targetState))
            self.search_pq.sort(key=lambda x: x[0])


