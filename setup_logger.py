import logging
from datetime import date
import os

if 'logs' not in os.listdir():
    os.mkdir('logs')

logging.basicConfig(format='%(name)s,%(levelname)s,%(message)s',filename=f'logs/{date.today().strftime("%d-%m-%Y")}.log')
logger = logging.getLogger('timer')
logger.setLevel(logging.INFO) #set the minimum level of message logging