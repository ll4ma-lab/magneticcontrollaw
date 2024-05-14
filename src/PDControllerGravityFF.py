import numpy as np
from PDController import PDController

class PDControllerGravityFF(PDController):
    def get_feedforward_component(self,current,desired):
        gravity = np.zeros((6,1))
        gravity[2,0] = 0.000014

        return gravity