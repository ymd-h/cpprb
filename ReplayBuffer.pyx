# distutils: language = c++

from ReplayBuffer cimport ReplayBuffer as cppReplayBuffer

class ReplayBuffer:
    def __init__(self):
        print("Hello World")

if __name__ is "main":
    rb = ReplayBuffer()
