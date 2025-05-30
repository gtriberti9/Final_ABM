# This file is no longer needed as agents are now defined in model.py
# This is kept for compatibility with import statements

# Import the agent classes from model.py
from model import CentralBank as CentralBankAgent
from model import CommercialBank as CommercialBankAgent  
from model import Firm as FirmAgent
from model import Consumer as ConsumerAgent

# For backward compatibility, you can still import agents from this file
# but they are actually defined in model.py now