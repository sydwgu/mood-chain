# shared memory module to live update the 'last_emotion' variable across scripts in directory

from multiprocessing import Manager

# Create a manager and shared dictionary
_manager = Manager()
shared_data = _manager.dict()

# Initialize with a default value
shared_data['last_emotion'] = None
