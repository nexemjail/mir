import multiprocessing
from sampler import Sampler
import numpy as np


def convert_with_processes(file_path, duration=30.0, half_part_length=0.1,
                           offset = 30, num_processes=multiprocessing.cpu_count()):
    global song

    song = Sampler(file_path, duration=duration, offset = offset)

    task_queue = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()

    parts = song.split(half_part_length)
    part_arr = [np.append(parts[i-1], parts[i]) for i in xrange(1, len(parts))]

    for element in part_arr:
        task_queue.put(element)
    for _ in xrange(num_processes):
        task_queue.put(None)

    tasks = []
    for _ in xrange(num_processes):
        process = QueueProcess(task_queue, results, take_feature)
        tasks.append(process)
        process.start()

    task_queue.join()

    result_array = []
    for i in xrange(len(part_arr)):
            result_array.append(results.get())
    result_array = np.asarray(result_array)
    return result_array


def take_feature(part):
    sample = Sampler(part, sample_rate=song.sample_rate)
    sample.compute_features()
    feature = sample.extract_features()
    return feature


class QueueProcess(multiprocessing.Process):
    def __init__(self, task_queue, result_queue, function):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.function = function

    def run(self):
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                self.task_queue.task_done()
                break
            processed_value = self.function(next_task)
            self.task_queue.task_done()
            self.result_queue.put(processed_value)
