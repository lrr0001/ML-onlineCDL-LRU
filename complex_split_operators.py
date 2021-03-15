from tensorflow.python.ops import math_ops as mathops

def square(x):
    return mathops.complex(mathops.real(x)*mathops.real(x),mathops.imag(x)*mathops.imag(x))

def sqrt(x):
    return mathops.complex(mathops.sqrt(mathops.real(x)),mathops.sqrt(mathops.imag(x)))

def divide(x,y):
    return mathops.complex(mathops.real(x)/mathops.real(y),mathops.imag(x)/mathops.imag(y))

def safe_divide(x,y,eps):
    return mathops.complex(mathops.real(x)/(mathops.real(y) + eps),maathops.imag(x)/(mathops.imag(y) + eps))

def realmaximum(x,y):
    return mathops.maximum(x,mathops.maximum(mathops.real(y),mathops.imag(y)))

