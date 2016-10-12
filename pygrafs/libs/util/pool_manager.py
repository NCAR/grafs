from time import sleep
import sys


def pool_manager(procs, return_results=True):
    completed = []
    results = {}
    while len(procs) > 0:
        sleep(0.1)
        for name, proc in procs.iteritems():
            if proc.ready():
                if proc.successful():
                    sys.stdout.write("\r{0} completed".format(name))
                else:
                    print(name, " failed")
                if return_results:
                    results[name] = proc.get()
                else:
                    proc.get()
                completed.append(name)
        while len(completed) > 0:
            del procs[completed.pop()]
    if return_results:
        return results
    else:
        return 0
