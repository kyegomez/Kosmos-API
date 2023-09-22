from exa import Kosmos
from exa.utils import Deploy

model = Kosmos()

api = Deploy(model)
api.run()