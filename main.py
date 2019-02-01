import random


class Map(object):
    def __init__(self, size):
        """Generates a board with dimensions size X size"""
        self.size = size
        self.matrix = []
        # Equal probabilities in the beginning. This represents the probability
        # of having target.
        self.probabilities = [[1 / (size * size)
                               for i in range(size)] for j in range(size)]
        # Initializing with 0s. This represents the probabilities of finding
        # the target.
        self.prob_of_finding = [[0 for i in range(size)] for j in range(size)]
        # Creating the map with the specified distribution of flat, hilly,
        # forested or maze cells
        for i in range(size):
            row = []
            for j in range(size):
                x = random.random()
                if x < 0.2:  # Cell is flat
                    row.append(0)
                elif x < 0.5:   # Cell is hilly
                    row.append(1)
                elif x < 0.8:   # Cell is forested
                    row.append(2)
                else:   # Cell is maze of caves
                    row.append(3)
            self.matrix.append(row)
        self.update_prob_of_finding()
        # Randomly choosing the target cell
        self.target_x = random.randint(0, size - 1)
        self.target_y = random.randint(0, size - 1)
        self.target_type = self.cell_type(
            self.target_x, self.target_y)  # Target cell type
        self.query_count = 0    # To keep track of the number of queries
        self.total_cost = 0
        self.cost_of_travel = 0

    def set_new_target(self):
        self.target_x = random.randint(0, self.size - 1)
        self.target_y = random.randint(0, self.size - 1)
        self.target_type = self.matrix[self.target_x][self.target_y]

    def reset_query_count(self):
        self.query_count = 0
        self.cost_of_travel = 0
        self.total_cost = 0
        self.probabilities = [[1 / (size * size)
                        for i in range(size)] for j in range(size)]
        self.prob_of_finding = [[0 for i in range(size)] for j in range(size)]
        self.update_prob_of_finding()

    def cell_type(self, x, y):
        """Returns the cell type at position (x, y)"""
        return self.matrix[x][y]

    def is_false_negative(self):
        """Returns True with probability P(Target not found in cell|Target being in cell) 
        according to the probabilities stated in the problem"""
        x = random.random()
        if self.target_type == 0:
            return x < 0.1
        elif self.target_type == 1:
            return x < 0.3
        elif self.target_type == 2:
            return x < 0.7
        else:
            return x < 0.9

    def update_probs(self, x, y):
        """Updates the probability for the entire map given that querying 
        the cell at position (x, y) returned False"""
        t = self.cell_type(x, y)
        if t == 0:
            f = 0.1
        elif t == 1:
            f = 0.3
        elif t == 2:
            f = 0.7
        else:
            f = 0.9
        prob_of_failure = f * \
            self.probabilities[x][y] + (1 - self.probabilities[x][y])
        self.probabilities[x][y] = f * \
            self.probabilities[x][y] / prob_of_failure
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) != (x, y):
                    self.probabilities[i][j] = self.probabilities[i][j] / \
                        prob_of_failure

    def update_prob_of_finding(self):
        """Updates the probabilities of finding the target given the probabilties 
        of having the target for each cell in the board"""
        for i in range(self.size):
            for j in range(self.size):
                t = self.cell_type(i, j)
                p = [0.9, 0.7, 0.3, 0.1][t]
                self.prob_of_finding[i][j] = p * self.probabilities[i][j]

    def query(self, x, y):
        """Returns True if the target is found after querying the cell at position (x, y), 
        otherwise updates the probabilities for the entire board and returns False"""
        self.query_count += 1
        if x == self.target_x and y == self.target_y and not self.is_false_negative():
            return True
        else:
            self.update_probs(x, y)
            self.update_prob_of_finding()
            return False

    def calculate_heuristic(self, x, y):
        """ Returns the heuristic value for the cell at position (x, y)"""
        s = 0
        for i in range(self.size):
            for j in range(self.size):
                s += self.probabilities[i][j] / (1 + abs(x - i) + abs(y - j))
        return s
    
    def update_travel_cost(self, current_position, previous_position):
        """ Updates the cost of travel from the previous position to the current """
        prev_x, prev_y = previous_position
        curr_x, curr_y = current_position

        self.cost_of_travel += abs(prev_x-curr_x) + abs(prev_y-curr_y)

    def find_max_pos(self, is_prob_of_finding=False):
        """Returns the position of the cell with the maximum probability. 
        If is_prob_of_finding=True, it considers the maximum probability of 
        finding the target, otherwise it considers the maximum probability 
        of having the target."""
        if is_prob_of_finding:
            max_val = max(max(self.prob_of_finding, key=lambda k: max(k)))
            for i in range(self.size):
                for j in range(self.size):
                    if self.prob_of_finding[i][j] == max_val:
                        return (i, j)
        max_val = max(max(self.probabilities, key=lambda k: max(k)))
        for i in range(self.size):
            for j in range(self.size):
                if self.probabilities[i][j] == max_val:
                    return (i, j)

    def find_max_heuristic(self, x, y, htype):

        """ Returns the position of the adjacent cell with the
        maximum heuristic value or maximum probability of finding the target """

        adjacent_cells = [(x-1, y), (x, y+1), (x+1, y), (x, y-1)]

        if htype == 'heuristic':
            max_heuristic = 0
            max_i, max_j = x, y
            for cell in adjacent_cells:
                i, j = cell[0], cell[1]
                if (0 <= i <= self.size-1) and (0 <= j <= self.size-1):
                    current_heuristic = self.calculate_heuristic(i, j)
                    if current_heuristic > max_heuristic:
                        max_heuristic = current_heuristic
                        max_i, max_j = i, j

        elif htype == 'greedy':
            max_prob = 0
            max_i, max_j = x, y
            for cell in adjacent_cells:
                i, j = cell[0], cell[1]
                if (0 <= i <= self.size-1) and (0 <= j <= self.size-1):
                    current_prob = self.prob_of_finding[i][j]
                    if current_prob > max_prob:
                        max_prob = current_prob
                        max_i, max_j = i, j
                
        return max_i, max_j


if __name__ == '__main__':

    size = 20
    number_of_maps = 1
    number_of_targets = 10
    threshold = 1/(size*size)

    """ QUESTION 3 """
    
    for h in [True, False]:
        print(f"[is_prob_of_finding = {h}]")
        for i in range(number_of_maps): # Testing for 10 different maps
            arr = []
            cost =  []
            m = Map(size)  # Map of size 50 X 50
            previous_position = None
            for j in range(number_of_targets): # for fixed map, 10 different target locations
                while True:
                    # Set is_prob_of_finding=False for rule-1 and True for rule-2
                    x, y = m.find_max_pos(is_prob_of_finding=h)
                    if previous_position:
                        m.update_travel_cost([x,y], previous_position)
                    if m.query(x, y):
                        m.total_cost = m.query_count+m.cost_of_travel
                        break
                    previous_position = [x, y]
                arr.append(m.query_count)
                cost.append(m.total_cost)
                print(f"\t{i+1}.{j+1}. Query Count: {m.query_count}, Total Cost: {m.total_cost}")
                m.set_new_target()
                m.reset_query_count()
            print(f"\tAverage number of queries {i+1}: {sum(arr) / len(arr)}")
            print(f"\tAverage total cost {i+1}: {sum(cost) / len(cost)}")

    """ QUESTION 4 """
    
    print("[Using Heuristic]")
    for i in range(number_of_maps): # Testing for 10 different maps
        arr = []
        m = Map(size)  # Map of size 50 X 50
        x, y = m.find_max_pos(is_prob_of_finding=True)
        previous_position = None
        for j in range(number_of_targets): # for fixed map, 10 different target locations
            while True:
                # if m.probabilities[x][y] >= threshold:
                if previous_position:
                    m.update_travel_cost([x,y], previous_position)
                if m.query(x, y):
                    m.total_cost = m.query_count+m.cost_of_travel
                    break
                previous_position = [x, y]
                x, y = m.find_max_heuristic(x, y,'heuristic')
            arr.append(m.query_count)
            cost.append(m.total_cost)
            print(f"\t{i+1}.{j+1}. Query Count: {m.query_count}, Total Cost: {m.total_cost}")
            m.set_new_target()
            m.reset_query_count()
        print(f"\tAverage number of queries {i+1}: {sum(arr) / len(arr)}")
        print(f"\tAverage total cost {i+1}: {sum(cost) / len(cost)}")