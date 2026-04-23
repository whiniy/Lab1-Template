from model import (
    Location,
    Portal,
    Wizard,
    Goblin,
    Crystal,
    WizardMoves,
    GoblinMoves,
    GameAction,
    GameState,
)
from agents import ReasoningWizard
from dataclasses import dataclass


class WizardGreedy(ReasoningWizard):
    def evaluation(self, state: GameState) -> float:
        wizard_loc = state.get_all_entity_locations(Wizard)[0]
        portal_loc = state.get_all_tile_locations(Portal)[0]
        goblin_locs = state.get_all_entity_locations(Goblin)

        def manhat(loc1: Location, loc2: Location) -> float:
            return abs(loc1.row - loc2.row) + abs(loc1.col - loc2.col)
        
        # winning state
        if wizard_loc == portal_loc:
            return float('inf')
        
        portal_distance = manhat(wizard_loc, portal_loc)

        # getting to portal is more important than avoiding goblins, so weight it more heavily
        score = -portal_distance * 5
        if goblin_locs:
            closest_goblin_distance = min(manhat(wizard_loc, goblin_loc) for goblin_loc in goblin_locs)
            # the farther away the goblins are, the better, so add to score
            score += closest_goblin_distance * 2

            # decrease score based on how close the closest goblin is
            if closest_goblin_distance == 0:
                return float('-inf')
            if closest_goblin_distance == 1:
                score -= 50
            if closest_goblin_distance == 2:
                score -= 15

        return score


class WizardMiniMax(ReasoningWizard):
    max_depth: int = 2

    def evaluation(self, state: GameState) -> float:
        wizard_locs = state.get_all_entity_locations(Wizard)
        if not wizard_locs:
            return float("-inf")

        wizard_loc = wizard_locs[0]

        portal_loc = state.get_all_tile_locations(Portal)[0]
        goblin_locs = state.get_all_entity_locations(Goblin)

        def manhat(loc1: Location, loc2: Location) -> float:
            return abs(loc1.row - loc2.row) + abs(loc1.col - loc2.col)
        
        # winning state
        if wizard_loc == portal_loc:
            return float('inf')
        
        portal_distance = manhat(wizard_loc, portal_loc)

        # getting to portal is more important than avoiding goblins, so weight it more heavily
        score = -portal_distance * 8
        for goblin_loc in goblin_locs:
            goblin_distance = manhat(wizard_loc, goblin_loc)

            # decrease score based on how close the closest goblin is
            if goblin_distance == 0:
                return float('-inf')
            if goblin_distance == 1:
                score -= 110
            if goblin_distance == 2:
                score -= 50
            if goblin_distance == 3:
                score -= 15
            
            score += 1

        return score

    def is_terminal(self, state: GameState) -> bool:
        # if there are no wizards 
        if not state.get_all_entity_locations(Wizard):
            return True

        wizard_loc = state.get_all_entity_locations(Wizard)[0]
        portal_loc = state.get_all_tile_locations(Portal)[0]

        # if wizard has reached portal
        if wizard_loc == portal_loc:
            return True
        
        return False

    def react(self, state: GameState) -> WizardMoves:
        # get all successor states from current state
        successors = self.get_successors(state)

        # keep track of best move and best minimax value
        best_action = None
        best_value = float("-inf")

        # for every possible action and resulting state for the wizard 
        for action, result in successors:
            value = self.minimax(result, 1)

            # reassign best action and best minimax value
            if value > best_value:
                best_action = action
                best_value = value
        
        # return action with greatest minimax value
        return best_action

    def minimax(self, state: GameState, depth: int):
        # check if gave is over or if we've reached max depth
        if self.is_terminal(state):
            return self.evaluation(state)
        
        # get all successor states from current state
        successors = self.get_successors(state)
        if not successors:
            return self.evaluation(state)

        # get active entity to determine if we are maximizing or minimizing
        active_ent = state.get_active_entity()
        # if Wizard, maximize value, else minimize value
        if isinstance(active_ent, Wizard):
            if depth == self.max_depth:
                return self.evaluation(state)
            value = float("-inf")
            for action, result in successors:
                value = max(value, self.minimax(result, depth + 1))
            return value
        else:
            value = float("inf")
            for action, result in successors:
                value = min(value, self.minimax(result, depth))
            return value


class WizardAlphaBeta(ReasoningWizard):
    max_depth: int = 2

    def evaluation(self, state: GameState) -> float:
        wizard_locs = state.get_all_entity_locations(Wizard)
        if not wizard_locs:
            return float("-inf")
        
        wizard_loc = wizard_locs[0]
        portal_loc = state.get_all_tile_locations(Portal)[0]
        goblin_locs = state.get_all_entity_locations(Goblin)
        crystal_locs = state.get_all_entity_locations(Crystal)

        def manhat(loc1: Location, loc2: Location) -> float:
            return abs(loc1.row - loc2.row) + abs(loc1.col - loc2.col)
        
        score = 0
        
        # winning state
        # take into account number of crystals left to determine score
        if wizard_loc == portal_loc:
            return 100000 - len(crystal_locs) * 100
        
        portal_distance = manhat(wizard_loc, portal_loc)
        # getting to portal is more important than avoiding goblins
        score -= portal_distance * 7

        if crystal_locs:
            closest_crystal_distance = min(manhat(wizard_loc, crystal_loc) for crystal_loc in crystal_locs)
            score = score - (closest_crystal_distance * 5) - (len(crystal_locs) * 30)
        
        # if goblins, farther away the better
        for goblin_loc in goblin_locs:
            goblin_distance = manhat(wizard_loc, goblin_loc)                
            
            # decrease score based on how close the closest goblin is
            if goblin_distance == 0:
                return float('-inf')
            if goblin_distance == 1:
                score -= 110
            if goblin_distance == 2:
                score -= 50
            if goblin_distance == 3:
                score -= 15
            score += 1
            
        return score

    def is_terminal(self, state: GameState) -> bool:
        # wizard dead
        if not state.get_all_entity_locations(Wizard):
            return True
        wizard_loc = state.get_all_entity_locations(Wizard)[0]
        portal_loc = state.get_all_tile_locations(Portal)[0]

        # wizard reached portal
        if wizard_loc == portal_loc:
            return True
        return False

    def react(self, state: GameState) -> WizardMoves:
        successors = list(self.get_successors(state))
        
        # larger alpha beta value, better for wizard
        successors.sort(key=lambda x: self.evaluation(x[1]), reverse=True)
        best_action = successors[0][0]
        best_value = float("-inf")
        
        alpha = float("-inf")
        beta = float("inf")

        for action, result in successors:
            self.alphaPass = alpha
            self.betaPass = beta
            value = self.alpha_beta_minimax(result, 1)

            if value > best_value:
                best_action = action
                best_value = value
            alpha = max(alpha, value)

            if alpha >= beta:
                break
        
        return best_action

    def alpha_beta_minimax(self, state: GameState, depth: int):
        def is_maximizing(state: GameState, depth: int, alpha: float, beta: float) -> float:
            # if game is over or we've reached max depth
            if self.is_terminal(state):
                return self.evaluation(state)
            
            # get all successor states from current state
            successors = list(self.get_successors(state))
            if not successors:
                return self.evaluation(state)
            
            active_ent = state.get_active_entity()

            # maximizing for wizard
            if isinstance(active_ent, Wizard):
                if depth == self.max_depth:
                    return self.evaluation(state)
                
                successors.sort(key=lambda x: self.evaluation(x[1]), reverse=True)
                value = float("-inf")
                for action, result in successors:
                    value = max(value, is_maximizing(result, depth + 1, alpha, beta))
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
                
                return value
            # minimizing for goblin
            else:
                successors.sort(key=lambda x: self.evaluation(x[1]))
                value = float("inf")
                for action, result in successors:
                    value = min(value, is_maximizing(result, depth, alpha, beta))
                    beta = min(beta, value)
                    if alpha >= beta:
                        break
                return value
        alpha = getattr(self, "alphaPass", float("-inf"))
        beta = getattr(self, "betaPass", float("inf"))
        return is_maximizing(state, depth, alpha, beta)
    

class WizardExpectimax(ReasoningWizard):
    max_depth: int = 2

    def evaluation(self, state: GameState) -> float:
        # get location of wizard, portal, goblins, crystals
        wizard_locs = state.get_all_entity_locations(Wizard)
        if not wizard_locs:
            return float("-inf")
        wizard_loc = wizard_locs[0]
        portal_loc = state.get_all_tile_locations(Portal)[0]
        goblin_locs = state.get_all_entity_locations(Goblin)
        crystal_locs = state.get_all_entity_locations(Crystal)

        def manhat(loc1: Location, loc2: Location) -> float:
            return abs(loc1.row - loc2.row) + abs(loc1.col - loc2.col)
        
        score = 0

        # if wizard reached portal
        if wizard_loc == portal_loc:
            return 100000 - len(crystal_locs) * 100
        
        portal_distance = manhat(wizard_loc, portal_loc)
        # getting to portal is more important than avoiding goblins
        score -= portal_distance * 5

        if crystal_locs:
            closest_crystal_distance = min(manhat(wizard_loc, crystal_loc) for crystal_loc in crystal_locs)
            score = score - (closest_crystal_distance * 3) - (len(crystal_locs) * 15)
        
        # if goblins, farther away the better
        if goblin_locs:
            closest_goblin_distance = min(manhat(wizard_loc, goblin_loc) for goblin_loc in goblin_locs)
            
            score += closest_goblin_distance * 7
            # decrease score based on how close the closest goblin is
            if closest_goblin_distance == 0:
                return float('-inf')
            if closest_goblin_distance == 1:
                score -= 50
            if closest_goblin_distance == 2:
                score -= 15

        return score

    def is_terminal(self, state: GameState) -> bool:
        # wizard dead
        if not state.get_all_entity_locations(Wizard):
            return True
        wizard_loc = state.get_all_entity_locations(Wizard)[0]
        portal_loc = state.get_all_tile_locations(Portal)[0]

        # wizard reached portal
        if wizard_loc == portal_loc:
            return True

    def react(self, state: GameState) -> WizardMoves:
        # get successor states from current state
        successors = self.get_successors(state)

        best_action = successors[0][0]
        best_value = float("-inf")

        for action, result in successors:
            value = self.expectimax(result, 1)

            if value > best_value:
                best_action = action
                best_value = value
        
        return best_action


    def expectimax(self, state: GameState, depth: int):
        # if game is over or we've reached max depth
        if self.is_terminal(state) or depth == self.max_depth:
            return self.evaluation(state)
        
        # get all successor states from current state
        successors = self.get_successors(state)

        if not successors:
            return self.evaluation(state)
        
        active_ent = state.get_active_entity()

        # maximizing for wizard
        if isinstance(active_ent, Wizard):
            value = float("-inf")
            for action, result in successors:
                value = max(value, self.expectimax(result, depth + 1))
            return value
        # expected value for goblin
        else:
            value = 0
            for action, result in successors:
                value += self.expectimax(result, depth)
            return value / len(successors)
