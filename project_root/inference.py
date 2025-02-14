from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from pydantic import conint
from config import config
import pickle
import os
import numpy as np
import math
import uvicorn
from sklearn.preprocessing import PolynomialFeatures