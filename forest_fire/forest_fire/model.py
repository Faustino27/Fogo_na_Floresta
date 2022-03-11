from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import Grid
from mesa.time import RandomActivation
from mesa.batchrunner import BatchRunner
from datetime import datetime

from .agent import TreeCell

def num_state(model,state):
    #print(dir(model.grid))
    return sum(1 for a in model.schedule.agents if a.condition is state)

def num_fine(model):
    return num_state(model,"Fine")

def num_onFire(model):
    return num_state(model,"On Fire")

def num_burned(model):
    return num_state(model,"Burned")

def num_survivor(model):
    return num_state(model,"Survivor")

class ForestFire(Model):
    """
    Simple Forest Fire model.
    """

    def __init__(self, width=100, height=100, density=0.65, surival_factor = 0.1):
        """
        Create a new forest fire model.

        Args:
            width, height: The size of the grid to model
            density: What fraction of grid cells have a tree in them.
        """
        # Set up model objects
        self.schedule = RandomActivation(self)
        self.grid = Grid(width, height, torus=False)

        self.datacollector = DataCollector(
            {
                "Fine": lambda m: self.count_type(m, "Fine"),
                "On Fire": lambda m: self.count_type(m, "On Fire"),
                "Burned Out": lambda m: self.count_type(m, "Burned Out"),
                "Survivor": lambda m: self.count_type(m, "Survivor"),
            }
        )

        # Place a tree in each cell with Prob = density
        for (contents, x, y) in self.grid.coord_iter():
            if self.random.random() < density:
                # Create a tree
                new_tree = TreeCell((x, y), self, surival_factor)
                # Set all trees in the first column on fire.
                if x == 0:
                    new_tree.condition = "On Fire"
                self.grid._place_agent((x, y), new_tree)
                self.schedule.add(new_tree)
                self.surival_factor = surival_factor

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        """
        Advance the model by one step.
        """
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

        # Halt if no more fire
        if self.count_type(self, "On Fire") == 0:
            self.running = False

    @staticmethod
    def count_type(model, tree_condition):
        """
        Helper method to count trees in a given condition in a given model.
        """
        count = 0
        for tree in model.schedule.agents:
            if tree.condition == tree_condition:
                count += 1
        return count

def batch_run():
    fix_params = {
        "width":100,
        "height": 100,
    }
    variable_params = {
        "surival_factor": [0.01, 0.1, 0.3],
        "density": [0.5, 0.65, 0.8]
    }

    batch_run = BatchRunner(
        ForestFire,
        variable_parameters = variable_params,
        fixed_parameters = fix_params,
        iterations = 10,
        max_steps = 50,
        model_reporters = {
            "Fine": num_fine,
            "On Fire": num_onFire,
            "Burned Out": num_burned,
            "Survivor": num_survivor
        }
    )

    max_steps_per_simulation = 10
    experiments_per_parameter_configuration = 10

    batch_run.run_all()
    run_model_data = batch_run.get_model_vars_dataframe()
#    run_agent_data = batch_run.get_agent_vars_dataframe()

    now = str(datetime.now())
    file_name_suffix =  ('_iter_'+str(experiments_per_parameter_configuration)+
                        '_steps_'+str(max_steps_per_simulation)+'_'+
                        now)
    run_model_data.to_csv('model_data'+'.csv')
    #run_agent_data.to_csv('agent_data'+file_name_suffix+'.csv')

