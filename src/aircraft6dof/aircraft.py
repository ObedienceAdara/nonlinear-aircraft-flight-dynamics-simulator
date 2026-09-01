from .equations import state_derivative
class AircraftModel:
    def __init__(self,parameters): self.parameters=parameters
    def derivative(self,state,controls,environment): return state_derivative(state,controls,environment,self.parameters)
